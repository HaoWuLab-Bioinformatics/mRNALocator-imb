import warnings
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, precision_recall_fscore_support,
    balanced_accuracy_score, roc_auc_score
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import random

warnings.filterwarnings("ignore")

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha if alpha is not None else torch.ones(5)
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)
        pt = torch.exp(-ce_loss)
        alpha = self.alpha[targets].to(device)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    if per_image:
        return lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    else:
        return lovasz_softmax_flat(probas.view(-1, probas.size(-1)), labels.view(-1), classes=classes)

def lovasz_softmax_flat(probas, labels, classes='present'):
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0).to(probas.device)

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

class LovaszSoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        probas = torch.softmax(logits, dim=1)
        return lovasz_softmax(probas, targets, classes='all')

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.from_numpy(m_list).float().to(device)
        self.m_list = m_list
        self.s = s
        self.weight = weight

    def forward(self, logits, targets):
        index = torch.zeros_like(logits, dtype=torch.uint8)
        index.scatter_(1, targets.data.view(-1, 1), 1)
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        logits_m = logits - batch_m
        return nn.CrossEntropyLoss(weight=self.weight)(self.s * logits_m, targets)

def read_fasta_dataset(file):
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

TRAIN_FASTA_PATHS = [
    '../data/train/Cytoplasm.fasta',
    '../data/train/Endoplasmic_reticulum.fasta',
    '../data/train/Extracellular_region.fasta',
    '../data/train/Mitochondria.fasta',
    '../data/train/Nucleus.fasta'
]

LABEL_MAP = {
    0: 'Cytoplasm',
    1: 'Endoplasmic_reticulum',
    2: 'Extracellular_region',
    3: 'Mitochondria',
    4: 'Nucleus'
}
NUM_CLASSES = len(LABEL_MAP)

def load_train_data(fasta_paths, feature_path):
    train_labels = []
    for file in fasta_paths:
        seq = read_fasta_dataset(file)
        if 'Cytoplasm' in file:
            label = [0] * len(seq)
        elif 'Endoplasmic' in file:
            label = [1] * len(seq)
        elif 'Extracellular' in file:
            label = [2] * len(seq)
        elif 'Mitochondria' in file:
            label = [3] * len(seq)
        elif 'Nucleus' in file:
            label = [4] * len(seq)
        train_labels += label
    train_labels = np.array(train_labels, dtype=int)
    
    train_feature = pd.read_csv(
        feature_path, sep=',', header=None, index_col=None, low_memory=False
    ).values.tolist()
    train_feature = np.array(train_feature)[1:, 1:]
    train_feature = np.array(train_feature, dtype=float)
    
    if len(train_labels) != train_feature.shape[0]:
        raise ValueError(f"Training label count ({len(train_labels)}) does not match feature count ({train_feature.shape[0]})!")
    
    unique, counts = np.unique(train_labels, return_counts=True)
    cls_num_list = [counts[np.where(unique == i)[0][0]] if i in unique else 0 for i in range(NUM_CLASSES)]
    return train_feature, train_labels, cls_num_list

device = torch.device('cuda' if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

BATCH_SIZE = 128
VAL_BATCH_SIZE = 512
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 4

class DNA_Transformer_Model(nn.Module):
    def __init__(self, input_dim=16, num_classes=5, d_model=16, nhead=2, num_layers=1, dim_feedforward=32):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model, bias=False)
        self.relu = nn.ReLU()
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.relu(x)
        x = self.transformer_encoder(x).squeeze(1)
        logits = self.classifier(x)
        return logits

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    loss_list = []
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        
        loss_list.append(loss.item() * batch_x.size(0))
    total_loss = sum(loss_list) / len(dataloader.dataset)
    return total_loss

def validate(model, dataloader, criterion, device, need_probs=False):
    model.eval()
    total_loss = 0.0
    all_preds_gpu = []
    all_labels_gpu = []
    all_probs_gpu = [] if need_probs else None
    loss_list = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            
            if scaler is not None:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
            else:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            
            loss_list.append(loss.item() * batch_x.size(0))
            all_preds_gpu.append(torch.argmax(logits, dim=1))
            all_labels_gpu.append(batch_y)
            if need_probs:
                all_probs_gpu.append(torch.softmax(logits, dim=1))
    
    total_loss = sum(loss_list) / len(dataloader.dataset)
    all_preds = torch.cat(all_preds_gpu).cpu().numpy()
    all_labels = torch.cat(all_labels_gpu).cpu().numpy()
    
    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    precision_cls, recall_cls, fscore_cls, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0)
    bacc = balanced_accuracy_score(all_labels, all_preds)
    auc = 0.0
    
    if need_probs and all_probs_gpu is not None:
        all_probs = torch.cat(all_probs_gpu).cpu().numpy()
        y_val_binarized = label_binarize(all_labels, classes=range(NUM_CLASSES))
        if NUM_CLASSES == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(y_val_binarized, all_probs, average='macro', multi_class='ovr')
    
    if need_probs:
        return (total_loss, acc, mcc, precision_macro, recall_macro, fscore_macro,
                precision_cls, recall_cls, fscore_cls, bacc, auc, all_preds, all_labels, all_probs)
    else:
        return (total_loss, acc, mcc, precision_macro, recall_macro, fscore_macro,
                precision_cls, recall_cls, fscore_cls, bacc, auc, all_preds, all_labels, None)

def init_model(num_classes=5, input_dim=16, loss_type="Cross Entropy (CE)", cls_num_list=None):
    model = DNA_Transformer_Model(input_dim=input_dim, num_classes=num_classes).to(device)
    
    cls_weights = None
    if cls_num_list is not None:
        cls_weights = 1.0 / np.array(cls_num_list)
        cls_weights = cls_weights / cls_weights.sum() * num_classes
        cls_weights = torch.from_numpy(cls_weights).float().to(device)
    
    if loss_type == "Cross Entropy (CE)":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == "Weighted Cross Entropy (WCE)":
        criterion = nn.CrossEntropyLoss(weight=cls_weights)
    elif loss_type == "Focal Loss":
        criterion = FocalLoss(alpha=cls_weights, gamma=2.0)
    elif loss_type == "Lovász-softmax Loss":
        criterion = LovaszSoftmaxLoss()
    elif loss_type == "LDAM Loss":
        criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, weight=cls_weights, s=30)
    else:
        raise ValueError(f"未知的损失函数类型: {loss_type}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    return model, criterion, optimizer, scheduler

def train_cv_evaluate(X_train, y_train, loss_type="Cross Entropy (CE)", cls_num_list=None, fold=10, random_seed=10):
    acc = np.zeros(fold, dtype=float)
    mcc = np.zeros(fold, dtype=float)
    precision = np.zeros(fold, dtype=float)
    recall = np.zeros(fold, dtype=float)
    fscore = np.zeros(fold, dtype=float)
    bacc = np.zeros(fold, dtype=float)
    auc = np.zeros(fold, dtype=float)
    class_precision = np.zeros((fold, NUM_CLASSES), dtype=float)
    class_recall = np.zeros((fold, NUM_CLASSES), dtype=float)
    class_fscore = np.zeros((fold, NUM_CLASSES), dtype=float)
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed) if torch.cuda.is_available() else None
    torch.cuda.manual_seed_all(random_seed) if torch.cuda.is_available() else None
    random.seed(random_seed)
    
    input_dim = X_train.shape[1]
    kf = KFold(n_splits=fold, shuffle=True, random_state=random_seed)
    
    i = 0
    for train_index, val_index in tqdm(kf.split(X_train), desc=f"{loss_type} - {fold}-Fold CV", total=fold, leave=True):
        X_tr_np, X_val_np = X_train[train_index], X_train[val_index]
        y_tr_np, y_val_np = y_train[train_index], y_train[val_index]
        
        X_train_fold_np, X_val_fold_np, y_train_fold_np, y_val_fold_np = train_test_split(
            X_tr_np, y_tr_np, test_size=1/9, random_state=random_seed+i, stratify=y_tr_np
        )
        
        X_train_fold = torch.from_numpy(X_train_fold_np).float()
        y_train_fold = torch.from_numpy(y_train_fold_np).long()
        X_val_fold = torch.from_numpy(X_val_fold_np).float()
        y_val_fold = torch.from_numpy(y_val_fold_np).long()
        X_test_fold = torch.from_numpy(X_val_np).float()
        y_test_fold = torch.from_numpy(y_val_np).long()
        
        train_tensor = TensorDataset(X_train_fold, y_train_fold)
        val_tensor = TensorDataset(X_val_fold, y_val_fold)
        test_tensor_fold = TensorDataset(X_test_fold, y_test_fold)
        
        train_loader = DataLoader(
            train_tensor, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True,
            persistent_workers=False
        )
        val_loader = DataLoader(
            val_tensor, 
            batch_size=VAL_BATCH_SIZE,
            shuffle=False, 
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=False
        )
        test_loader_fold = DataLoader(
            test_tensor_fold, 
            batch_size=VAL_BATCH_SIZE,
            shuffle=False, 
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=False
        )
        
        model, criterion, optimizer, scheduler = init_model(
            num_classes=5, input_dim=input_dim, loss_type=loss_type, cls_num_list=cls_num_list
        )
        best_val_acc = 0.0
        early_stop_count = 0
        best_model_state = None
        
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_mcc, val_p, val_r, val_f, _, _, _, val_bacc, val_auc, _, _, _ = validate(
                model, val_loader, criterion, device, need_probs=False
            )
            scheduler.step()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                early_stop_count = 0
            else:
                early_stop_count += 1
            
            if early_stop_count >= PATIENCE:
                break
        
        model.load_state_dict(best_model_state)
        _, test_acc, test_mcc, test_p, test_r, test_f, test_p_cls, test_r_cls, test_f_cls, test_bacc, test_auc, _, _, _ = validate(
            model, test_loader_fold, criterion, device, need_probs=True
        )
        
        acc[i] = test_acc
        mcc[i] = test_mcc
        precision[i] = test_p
        recall[i] = test_r
        fscore[i] = test_f
        bacc[i] = test_bacc
        auc[i] = test_auc
        class_precision[i] = test_p_cls
        class_recall[i] = test_r_cls
        class_fscore[i] = test_f_cls
        i += 1
    
    fold_metrics = {
        'ACC': acc, 'MCC': mcc, 'Precision': precision, 
        'Recall': recall, 'Fscore': fscore, 'BACC': bacc, 'AUC': auc
    }
    mean_metrics = {
        'ACC': np.mean(acc), 'MCC': np.mean(mcc),
        'Precision': np.mean(precision), 'Recall': np.mean(recall),
        'Fscore': np.mean(fscore), 'BACC': np.mean(bacc), 'AUC': np.mean(auc)
    }
    class_mean_metrics = {
        'Precision': np.mean(class_precision, axis=0),
        'Recall': np.mean(class_recall, axis=0),
        'Fscore': np.mean(class_fscore, axis=0)
    }
    
    return fold_metrics, mean_metrics, class_mean_metrics

def parse_feature_filename(file_name):
    name_without_ext = file_name[:-4]
    if '_' in name_without_ext:
        base_method = name_without_ext.split('_')[0]
        full_method = name_without_ext
    else:
        base_method = name_without_ext
        full_method = name_without_ext
    return base_method, full_method

def main():
    FEATURE_DIR = '../data/fusion/train/'
    file_name = "set2.csv"
    feature_path = os.path.join(FEATURE_DIR, file_name)
    fold = 10
    random_seed = 10
    
    print(f"Reading feature file: {feature_path}")
    print("Starting cross validation processing...\n")
    
    try:
        base_method, full_method = parse_feature_filename(file_name)
        
        X_train, y_train, cls_num_list = load_train_data(TRAIN_FASTA_PATHS, feature_path)
        
        loss_functions = [
            "Cross Entropy (CE)",
            "Weighted Cross Entropy (WCE)",
            "Focal Loss",
            "Lovász-softmax Loss",
            "LDAM Loss"
        ]
        all_results = {}
        
        print("\n" + "="*120)
        print("Testing loss functions (10-fold CV on training set)")
        print("="*120)
        
        for loss_type in loss_functions:
            print(f"\n{'='*80}")
            print(f"Processing {loss_type}...")
            print(f"{'='*80}")
            
            fold_metrics, mean_metrics, class_mean_metrics = train_cv_evaluate(
                X_train, y_train, loss_type=loss_type, cls_num_list=cls_num_list,
                fold=fold, random_seed=random_seed
            )
            
            all_results[loss_type] = {
                'fold_metrics': fold_metrics,
                'mean_metrics': mean_metrics,
                'class_mean_metrics': class_mean_metrics
            }
            
            print(f"\n{loss_type} - {full_method} - 10-Fold Cross Validation (Overall) Results")
            print("="*110)
            print(f"{'Fold':<8} {'ACC':<10} {'MCC':<10} {'Precision':<12} {'Recall':<10} {'Fscore':<10} {'BACC':<10} {'AUC':<10}")
            print("-"*110)
            for i in range(fold):
                print(f"{i+1:<8} {fold_metrics['ACC'][i]:<10.4f} {fold_metrics['MCC'][i]:<10.4f} "
                      f"{fold_metrics['Precision'][i]:<12.4f} {fold_metrics['Recall'][i]:<10.4f} "
                      f"{fold_metrics['Fscore'][i]:<10.4f} {fold_metrics['BACC'][i]:<10.4f} {fold_metrics['AUC'][i]:<10.4f}")
            print("-"*110)
            print(f"{'Mean':<8} {mean_metrics['ACC']:<10.4f} {mean_metrics['MCC']:<10.4f} "
                  f"{mean_metrics['Precision']:<12.4f} {mean_metrics['Recall']:<10.4f} "
                  f"{mean_metrics['Fscore']:<10.4f} {mean_metrics['BACC']:<10.4f} {mean_metrics['AUC']:<10.4f}")
            print("="*110 + "\n")
            
            print(f"{loss_type} - {full_method} - Per-Class Mean Results (10-Fold CV)")
            print("="*80)
            print(f"{'Class':<25} {'Precision':<12} {'Recall':<10} {'Fscore':<10}")
            print("-"*80)
            for label_id in sorted(LABEL_MAP.keys()):
                cls_name = LABEL_MAP[label_id]
                cls_prec = class_mean_metrics['Precision'][label_id]
                cls_rec = class_mean_metrics['Recall'][label_id]
                cls_f1 = class_mean_metrics['Fscore'][label_id]
                print(f"{cls_name:<25} {cls_prec:<12.4f} {cls_rec:<10.4f} {cls_f1:<10.4f}")
            print("="*80 + "\n")
        
        print("\n" + "="*120)
        print("Summary of all loss functions (10-fold CV mean)")
        print("="*120)
        print(f"{'Loss Function':<25} {'ACC':<10} {'MCC':<10} {'Fscore':<10} {'BACC':<10} {'AUC':<10}")
        print("-"*75)
        for loss_type, results in all_results.items():
            mean_metrics = results['mean_metrics']
            print(f"{loss_type:<25} {mean_metrics['ACC']:<10.4f} {mean_metrics['MCC']:<10.4f} "
                  f"{mean_metrics['Fscore']:<10.4f} {mean_metrics['BACC']:<10.4f} {mean_metrics['AUC']:<10.4f}")
        print("="*120 + "\n")
        
        print("Processing completed!")
    
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")

if __name__ == "__main__":
    main()