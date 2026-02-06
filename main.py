import os
import shutil
import warnings
from datetime import datetime

import numpy as np
import torch
from torch import optim

import Nets
import config
import dataset
import utils
import wandb

from train import Trainer

wandb.init(project="m_RNA_CNN ", entity="seaside")

warnings.filterwarnings("ignore")

train_file_name = ['./data/train/Cytoplasm_train.fasta', './data/train/Endoplasmic_reticulum_train.fasta',
                   './data/train/Extracellular_region_train.fasta', './data/train/Mitochondria_train.fasta',
                   './data/train/Nucleus_train.fasta']

test_file_name = ['./data/indep_1/Cytoplasm_indep1.fasta',
                  './data/indep_1/Endoplasmic_reticulum_indep1.fasta',
                  './data/indep_1/Extracellular_region_indep1.fasta',
                  './data/indep_1/Mitochondria_indep1.fasta',
                  './data/indep_1/Nucleus_indep1.fasta']


def main():
    args = config.parse_args()
    wandb.config.update(args)

    if torch.cuda.is_available():
        device = args.device
    else:
        device = 'cpu'

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    now_time = datetime.now()
    now_time_str = str(now_time.month) + str(now_time.day) + str(now_time.hour) + str(now_time.minute)

    model_path = os.path.join(
        args.save, now_time_str + '.pt')

    train_dataset = dataset.LDAMLncAtlasDataset(
        train_file_name, args.train_dataset)
    # dev_dataset = dataset.LDAMLncAtlasDataset(args.dev_CNRCI, args.dev_dataset)
    test_dataset = dataset.LDAMLncAtlasDataset(
        test_file_name, args.test_dataset)

    model = Nets.CNN_MLP(args)

    if device != 'cpu':
        model.cuda(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)

    best_fscore = 0

    # model.initialize()

    cls_num_list = train_dataset.get_cls_num_list()
    for epoch in range(args.epochs):
        idx = epoch // 799
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / \
                          np.sum(per_cls_weights) * len(cls_num_list)
        if device != 'cpu':
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(device)
        else:
            per_cls_weights = torch.FloatTensor(per_cls_weights)

        criterion = Nets.LDAMLoss(
            cls_num_list=cls_num_list,
            max_m=0.5,
            s=30,
            weight=per_cls_weights,
            device=args.device).cuda(device)
        # criterion = Nets.FocalLoss(gamma=1.5, alpha=None)

        trainer = Trainer(args, model, criterion, optimizer, device)

        # if os.path.exists(model_path + '.tmp') is False:
        #     _ = trainer.train(train_dataset)
        #
        #     torch.save(model.state_dict(), model_path + '.tmp')

        _ = trainer.train(train_dataset)

        train_loss, train_preds, train_labels = trainer.test(train_dataset)

        train_preds_CNRCI = train_preds
        train_labels_CNRCI = train_labels

        train_accuracy, train_precision, train_recall, train_report, train_support, train_report_dict = \
            utils.evaluate_metrics_sklearn(
                train_preds_CNRCI.numpy(),
                train_labels_CNRCI.numpy())
        #
        # dev_loss, dev_preds, dev_labels = trainer.test(dev_dataset)
        #
        # dev_preds_CNRCI = dev_preds
        # dev_labels_CNRCI = dev_labels
        #
        # dev_accuracy, dev_precision, dev_recall, dev_roc_auc, dev_report, dev_support, dev_report_dict = \
        #     utils.evaluate_metrics_sklearn(
        #         dev_preds_CNRCI.numpy(), dev_labels_CNRCI.numpy())

        test_loss, test_preds, test_labels = trainer.test(test_dataset)

        test_preds_CNRCI = test_preds
        test_labels_CNRCI = test_labels

        test_accuracy, test_precision, test_recall, test_report, test_support, test_report_dict = \
            utils.evaluate_metrics_sklearn(
                test_preds_CNRCI.numpy(), test_labels_CNRCI.numpy())

        print('Evaluation after epoch {}'.format(epoch + 1))
        print(
            "{:s}test: CNRCI classfication report:\n{:s}".format(
                '\n' + '-' * 80 + '\n',
                test_report,
                '\n' + '-' * 80
            ))

        wandb.log({"train_loss": train_loss, "test_loss": test_loss,
                   "train_ACC": train_accuracy,
                   "train_f1": train_report_dict['macro avg']['f1-score'],
                   "test_ACC": test_accuracy,
                   "test_f1": test_report_dict['macro avg']['f1-score'],
                   "epoch": epoch + 1})

        fscore = test_report_dict['weighted avg']['f1-score']

        if best_fscore < fscore:
            best_fscore = fscore
            wandb.run.summary["best_epoch"] = epoch
            wandb.run.summary["best_ACC"] = test_accuracy
            wandb.run.summary["best_fscore"] = best_fscore
            torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
