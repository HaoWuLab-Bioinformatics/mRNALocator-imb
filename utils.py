import gensim
import numpy as np
import torch
from sklearn import metrics

train_file_name = ['../data/train/Cytoplasm_train.fasta', '../data/train/Endoplasmic_reticulum_train.fasta',
                   '../data/train/Extracellular_region_train.fasta', '../data/train/Mitochondria_train.fasta',
                   '../data/train/Nucleus_train.fasta']

test_file_name = ['../data/train/Cytoplasm_train_indep1.fasta',
                  '../data/train/Endoplasmic_reticulum_train_indep1.fasta',
                  '../data/train/Extracellular_region_train_indep1.fasta',
                  '../data/train/Mitochondria_train_indep1.fasta',
                  '../data/train/Nucleus_train_indep1.fasta']


def read_fasta(file):
    f = open(file)
    documents = f.readlines()
    string = ""
    flag = 0
    fea = []
    for document in documents:
        if document.startswith(">") and flag == 0:
            flag = 1
            continue
        elif document.startswith(">") and flag == 1:
            string = string.upper()
            fea.append(string)
            string = ""
        else:
            string += document
            string = string.strip()
            string = string.replace(" ", "")
    fea.append(string)
    f.close()
    return fea


def collate_fn(data_list):
    feature = [i['feature'] for i in data_list]
    label = [i['label'] for i in data_list]
    return feature, label


def evaluate_metrics_sklearn(y_score, y_true, threshold=0.5):
    # y_true = y_true.astype(np.int)
    # y_pred = (y_score > threshold).astype(np.int)
    y_pred = y_score
    # roc_auc = metrics.roc_auc_score(y_true, y_score)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average=None)
    recall = metrics.recall_score(y_true, y_pred, average=None)
    report = metrics.classification_report(y_true, y_pred, digits=5)
    report_dict = metrics.classification_report(
        y_true, y_pred, output_dict=True)
    support = y_true.shape[0]
    # print(report)
    return accuracy, precision, recall, report, support, report_dict



def number_to_pattern(num, base):
    pattern = ""
    for i in range(base):
        div = num // 4 ** (base - i - 1)
        num = num - 4 ** (base - i - 1) * div
        if div == 0:
            pattern += "A"
        elif div == 1:
            pattern += "C"
        elif div == 2:
            pattern += "G"
        elif div == 3:
            pattern += "T"
        else:
            pass
    return pattern


def embed_from_pretrained(args):
    model = gensim.models.word2vec.Word2Vec.load('./save/' + args.cell_line + '_k=' + str(args.k) + '_vs=' + str(args.embed_num) + '.w2v')
    # model = gensim.models.word2vec.Word2Vec.load('./save/HUVEC.w2v')
    embed = torch.zeros(4 ** args.k, args.embed_num)
    for i in range(4 ** args.k):
        try:
            pattern = number_to_pattern(i, args.k)
            embed[i, :] = torch.Tensor(model.wv[pattern])
        except:
            print("Pattern {:s} not found.".format(pattern))
    return embed
