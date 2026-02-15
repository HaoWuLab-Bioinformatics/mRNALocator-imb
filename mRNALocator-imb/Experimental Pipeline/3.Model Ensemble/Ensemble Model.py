import warnings
import random
import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, precision_recall_fscore_support,
    balanced_accuracy_score, roc_auc_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from imblearn.over_sampling import SMOTE, RandomOverSampler

warnings.filterwarnings("ignore")

SEED = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) if torch.cuda.is_available() else None
torch.cuda.manual_seed_all(SEED) if torch.cuda.is_available() else None
random.seed(SEED)

TRAIN_DIR = "./data/train"
WEIGHT_DIR = "./data/weight"
FUSION_TRAIN_DIR = "./data/fusion/train"
FUSION_WEIGHT_DIR = "./data/fusion/weight"

TRAIN_FILES = [
    f'{TRAIN_DIR}/Cytoplasm.fasta',
    f'{TRAIN_DIR}/Endoplasmic_reticulum.fasta',
    f'{TRAIN_DIR}/Extracellular_region.fasta',
    f'{TRAIN_DIR}/Mitochondria.fasta',
    f'{TRAIN_DIR}/Nucleus.fasta'
]

WEIGHT_FILES = [
    f'{WEIGHT_DIR}/Cytoplasm.fasta',
    f'{WEIGHT_DIR}/Endoplasmic_reticulum.fasta',
    f'{WEIGHT_DIR}/Extracellular_region.fasta',
    f'{WEIGHT_DIR}/Mitochondria.fasta',
    f'{WEIGHT_DIR}/Nucleus.fasta'
]

RF_FEATURE_FILE = "set.csv"
TF_FEATURE_FILE = "set2.csv"
BATCH_SIZE = 128
VAL_BATCH_SIZE = 512
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 4
TEST_SIZE = 0.3
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

LABEL_MAP = {
    0: 'Cytoplasm',
    1: 'Endoplasmic_reticulum',
    2: 'Extracellular_region',
    3: 'Mitochondria',
    4: 'Nucleus'
}
NUM_CLASSES = len(LABEL_MAP)

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super().__init__()
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
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1,1))
        logits_m = logits - batch_m
        return nn.CrossEntropyLoss(weight=self.weight)(self.s * logits_m, targets)

def read_fasta_dataset(file):
    f = open(file)
    docs = f.readlines()
    string = ""
    flag = 0
    fea = []
    for doc in docs:
        if doc.startswith(">") and flag == 0:
            flag = 1
            continue
        elif doc.startswith(">") and flag == 1:
            string = string.upper()
            fea.append(string)
            string = ""
        else:
            string += doc
            string = string.strip()
            string = string.replace(" ", "")
    fea.append(string)
    f.close()
    return fea

def load_data(fasta_paths, feature_path, return_cls_num=False):
    labels = []
    for file in fasta_paths:
        seqs = read_fasta_dataset(file)
        if 'Cytoplasm' in file:
            labels += [0]*len(seqs)
        elif 'Endoplasmic' in file:
            labels += [1]*len(seqs)
        elif 'Extracellular' in file:
            labels += [2]*len(seqs)
        elif 'Mitochondria' in file:
            labels += [3]*len(seqs)
        elif 'Nucleus' in file:
            labels += [4]*len(seqs)
    labels = np.array(labels, dtype=int)
    
    feat = pd.read_csv(feature_path, header=None).values
    feat = np.array(feat[1:,1:], dtype=float)
    
    if len(labels) != feat.shape[0]:
        raise ValueError("Label and feature count mismatch")
    
    if return_cls_num:
        unique, cnt = np.unique(labels, return_counts=True)
        cls_num = [cnt[np.where(unique==i)[0][0]] if i in unique else 0 for i in range(NUM_CLASSES)]
        return feat, labels, cls_num
    return feat, labels

def load_combined_data(weight_train_idx=None, return_cls_num=False):
    Xr_train, y_train = load_data(TRAIN_FILES, os.path.join(FUSION_TRAIN_DIR, RF_FEATURE_FILE))
    Xt_train, _ = load_data(TRAIN_FILES, os.path.join(FUSION_TRAIN_DIR, TF_FEATURE_FILE))
    
    Xr_weight, y_weight = load_data(WEIGHT_FILES, os.path.join(FUSION_WEIGHT_DIR, RF_FEATURE_FILE))
    Xt_weight, _ = load_data(WEIGHT_FILES, os.path.join(FUSION_WEIGHT_DIR, TF_FEATURE_FILE))
    
    if weight_train_idx is not None:
        Xr_weight = Xr_weight[weight_train_idx]
        Xt_weight = Xt_weight[weight_train_idx]
        y_weight = y_weight[weight_train_idx]
    
    Xr_comb = np.concatenate([Xr_train, Xr_weight], axis=0)
    Xt_comb = np.concatenate([Xt_train, Xt_weight], axis=0)
    y_comb = np.concatenate([y_train, y_weight], axis=0)
    
    if return_cls_num:
        unique, cnt = np.unique(y_comb, return_counts=True)
        cls_num = [cnt[np.where(unique==i)[0][0]] if i in unique else 0 for i in range(NUM_CLASSES)]
        return Xr_comb, Xt_comb, y_comb, cls_num
    return Xr_comb, Xt_comb, y_comb

class DNA_Transformer_Model(nn.Module):
    def __init__(self, input_dim=16, num_classes=5, d_model=16, nhead=2, num_layers=1, dim_feedforward=32):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model, bias=False)
        self.relu = nn.ReLU()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model,16,bias=False), 
            nn.ReLU(), 
            nn.Linear(16,num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.relu(x)
        x = self.encoder(x).squeeze(1)
        return self.classifier(x)

def train_epoch(model, loader, crit, opt, device):
    model.train()
    total_loss = 0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def eval_tf(model, loader, crit, device, need_probs=False):
    model.eval()
    preds, lbls, probs = [], [], []
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            preds.append(torch.argmax(logits,dim=1).cpu().numpy())
            lbls.append(y.cpu().numpy())
            if need_probs:
                probs.append(torch.softmax(logits,dim=1).cpu().numpy())
    preds = np.concatenate(preds)
    lbls = np.concatenate(lbls)
    probs = np.concatenate(probs) if need_probs else None

    acc = accuracy_score(lbls, preds)
    mcc = matthews_corrcoef(lbls, preds)
    bacc = balanced_accuracy_score(lbls, preds)
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(lbls,preds,average='macro',zero_division=0)
    p_cls, r_cls, f_cls, _ = precision_recall_fscore_support(lbls,preds,average=None,zero_division=0)
    
    auc = 0.0
    if probs is not None:
        y_bin = label_binarize(lbls, classes=range(NUM_CLASSES))
        auc = roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
    
    return acc, mcc, bacc, p_mac, r_mac, f_mac, p_cls, r_cls, f_cls, auc, probs

def init_tf(input_dim, cls_num):
    model = DNA_Transformer_Model(input_dim=input_dim, num_classes=NUM_CLASSES).to(device)
    weights = 1.0 / np.array(cls_num)
    weights = weights / weights.sum() * NUM_CLASSES
    weights = torch.from_numpy(weights).float().to(device)
    crit = LDAMLoss(cls_num_list=cls_num, weight=weights)
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sch = StepLR(opt, step_size=10, gamma=0.5)
    return model, crit, opt, sch

def train_rf():
    print("Training RandomForest Base Model...")
    path = os.path.join(FUSION_TRAIN_DIR, RF_FEATURE_FILE)
    X, y = load_data(TRAIN_FILES, path)
    sm = SMOTE(random_state=SEED)
    Xr, yr = sm.fit_resample(X, y)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    with tqdm(total=100, desc="RF Training", unit="tree") as pbar:
        rf_model.fit(Xr, yr)
        pbar.update(100)
    
    return rf_model, X, y

def train_tf():
    print("Training Transformer Base Model...")
    path = os.path.join(FUSION_TRAIN_DIR, TF_FEATURE_FILE)
    X, y, cls_num = load_data(TRAIN_FILES, path, return_cls_num=True)
    ros = RandomOverSampler(random_state=SEED)
    Xr, yr = ros.fit_resample(X, y)
    Xt, Xv, yt, yv = train_test_split(Xr, yr, test_size=0.1, random_state=SEED, stratify=yr)
    
    tl = DataLoader(TensorDataset(torch.FloatTensor(Xt),torch.LongTensor(yt)), batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(TensorDataset(torch.FloatTensor(Xv),torch.LongTensor(yv)), batch_size=VAL_BATCH_SIZE)
    
    model, crit, opt, sch = init_tf(X.shape[1], cls_num)
    best_mcc, best_state, es = -1, None, 0
    
    with tqdm(total=EPOCHS, desc="TF Training", unit="epoch") as pbar:
        for e in range(EPOCHS):
            train_epoch(model, tl, crit, opt, device)
            res = eval_tf(model, vl, crit, device)
            sch.step()
            
            if res[1] > best_mcc:
                best_mcc = res[1]
                best_state = copy.deepcopy(model.state_dict())
                es = 0
            else:
                es +=1
                if es >= PATIENCE:
                    pbar.update(EPOCHS - e - 1)
                    break
            pbar.update(1)
    
    model.load_state_dict(best_state)
    return model, X.shape[1], cls_num, X, y

def load_weight_set():
    path_rf = os.path.join(FUSION_WEIGHT_DIR, RF_FEATURE_FILE)
    path_tf = os.path.join(FUSION_WEIGHT_DIR, TF_FEATURE_FILE)
    Xr, y = load_data(WEIGHT_FILES, path_rf)
    Xt, _ = load_data(WEIGHT_FILES, path_tf)
    return Xr, Xt, y

def search_fold_weights(rf, tf, input_dim, cls_num, Xr_val, Xt_val, y_val):
    tl = DataLoader(TensorDataset(torch.FloatTensor(Xt_val), torch.LongTensor(y_val)), batch_size=VAL_BATCH_SIZE)
    _, crit, _, _ = init_tf(input_dim, cls_num)
    
    prf = rf.predict_proba(Xr_val)
    ptf = eval_tf(tf, tl, crit, device, need_probs=True)[-1]
    
    best_mcc = -1
    best_alpha = 0.5
    weight_search_range = np.linspace(0.0, 1.0, 51)
    
    for alpha in weight_search_range:
        fused_probs = alpha * prf + (1 - alpha) * ptf
        pred = np.argmax(fused_probs, axis=1)
        mcc = matthews_corrcoef(y_val, pred)
        
        if mcc > best_mcc:
            best_mcc = mcc
            best_alpha = alpha
    
    return best_alpha

def search_fold_class_weights(rf, tf, input_dim, cls_num, Xr_val, Xt_val, y_val):
    from itertools import product
    tl = DataLoader(TensorDataset(torch.FloatTensor(Xt_val), torch.LongTensor(y_val)), batch_size=VAL_BATCH_SIZE)
    _, crit, _, _ = init_tf(input_dim, cls_num)
    
    prf = rf.predict_proba(Xr_val)
    ptf = eval_tf(tf, tl, crit, device, need_probs=True)[-1]
    
    best_mcc = -1
    best_class_alphas = np.ones(NUM_CLASSES) * 0.5
    weight_search_range = np.linspace(0.0, 1.0, 11)
    
    all_combinations = list(product(weight_search_range, repeat=NUM_CLASSES))
    with tqdm(total=len(all_combinations), desc="Class Weight Search", unit="comb") as pbar:
        for combo in all_combinations:
            class_alphas = np.array(combo)
            fused_probs = class_alphas * prf + (1 - class_alphas) * ptf
            pred = np.argmax(fused_probs, axis=1)
            mcc = matthews_corrcoef(y_val, pred)
            
            if mcc > best_mcc:
                best_mcc = mcc
                best_class_alphas = class_alphas.copy()
            pbar.update(1)
    return best_class_alphas

def eval_fold(rf, tf, input_dim, cls_num, alpha, Xr_test, Xt_test, y_test):
    prf = rf.predict_proba(Xr_test)
    pred_rf = np.argmax(prf, axis=1)
    acc_r = accuracy_score(y_test, pred_rf)
    mcc_r = matthews_corrcoef(y_test, pred_rf)
    bacc_r = balanced_accuracy_score(y_test, pred_rf)
    p_r, r_r, f_r, _ = precision_recall_fscore_support(y_test, pred_rf, average='macro', zero_division=0)
    p_cls_r, r_cls_r, f_cls_r, _ = precision_recall_fscore_support(y_test, pred_rf, average=None, zero_division=0)
    yb = label_binarize(y_test, classes=range(NUM_CLASSES))
    auc_r = roc_auc_score(yb, prf, average='macro', multi_class='ovr')
    
    tl = DataLoader(TensorDataset(torch.FloatTensor(Xt_test), torch.LongTensor(y_test)), batch_size=VAL_BATCH_SIZE)
    _, crit, _, _ = init_tf(input_dim, cls_num)
    acc_t, mcc_t, bacc_t, p_t, r_t, f_t, p_cls_t, r_cls_t, f_cls_t, auc_t, ptf = eval_tf(tf, tl, crit, device, need_probs=True)
    
    fused_probs = alpha * prf + (1 - alpha) * ptf
    pred_f = np.argmax(fused_probs, axis=1)
    
    acc_f = accuracy_score(y_test, pred_f)
    mcc_f = matthews_corrcoef(y_test, pred_f)
    bacc_f = balanced_accuracy_score(y_test, pred_f)
    p_f, r_f, f_f, _ = precision_recall_fscore_support(y_test, pred_f, average='macro', zero_division=0)
    p_cls_f, r_cls_f, f_cls_f, _ = precision_recall_fscore_support(y_test, pred_f, average=None, zero_division=0)
    auc_f = roc_auc_score(yb, fused_probs, average='macro', multi_class='ovr')
    
    return {
        "macro": {
            "rf": (acc_r, mcc_r, bacc_r, p_r, r_r, f_r, auc_r),
            "tf": (acc_t, mcc_t, bacc_t, p_t, r_t, f_t, auc_t),
            "fusion": (acc_f, mcc_f, bacc_f, p_f, r_f, f_f, auc_f)
        },
        "cls": {
            "rf": (p_cls_r, r_cls_r, f_cls_r),
            "tf": (p_cls_t, r_cls_t, f_cls_t),
            "fusion": (p_cls_f, r_cls_f, f_cls_f)
        }
    }

def eval_fold_class_weights(rf, tf, input_dim, cls_num, class_alphas, Xr_test, Xt_test, y_test):
    prf = rf.predict_proba(Xr_test)
    pred_rf = np.argmax(prf, axis=1)
    acc_r = accuracy_score(y_test, pred_rf)
    mcc_r = matthews_corrcoef(y_test, pred_rf)
    bacc_r = balanced_accuracy_score(y_test, pred_rf)
    p_r, r_r, f_r, _ = precision_recall_fscore_support(y_test, pred_rf, average='macro', zero_division=0)
    p_cls_r, r_cls_r, f_cls_r, _ = precision_recall_fscore_support(y_test, pred_rf, average=None, zero_division=0)
    yb = label_binarize(y_test, classes=range(NUM_CLASSES))
    auc_r = roc_auc_score(yb, prf, average='macro', multi_class='ovr')
    
    tl = DataLoader(TensorDataset(torch.FloatTensor(Xt_test), torch.LongTensor(y_test)), batch_size=VAL_BATCH_SIZE)
    _, crit, _, _ = init_tf(input_dim, cls_num)
    acc_t, mcc_t, bacc_t, p_t, r_t, f_t, p_cls_t, r_cls_t, f_cls_t, auc_t, ptf = eval_tf(tf, tl, crit, device, need_probs=True)
    
    fused_probs = class_alphas * prf + (1 - class_alphas) * ptf
    pred_f = np.argmax(fused_probs, axis=1)
    
    acc_f = accuracy_score(y_test, pred_f)
    mcc_f = matthews_corrcoef(y_test, pred_f)
    bacc_f = balanced_accuracy_score(y_test, pred_f)
    p_f, r_f, f_f, _ = precision_recall_fscore_support(y_test, pred_f, average='macro', zero_division=0)
    p_cls_f, r_cls_f, f_cls_f, _ = precision_recall_fscore_support(y_test, pred_f, average=None, zero_division=0)
    auc_f = roc_auc_score(yb, fused_probs, average='macro', multi_class='ovr')
    
    return {
        "macro": {
            "rf": (acc_r, mcc_r, bacc_r, p_r, r_r, f_r, auc_r),
            "tf": (acc_t, mcc_t, bacc_t, p_t, r_t, f_t, auc_t),
            "fusion_class": (acc_f, mcc_f, bacc_f, p_f, r_f, f_f, auc_f)
        },
        "cls": {
            "rf": (p_cls_r, r_cls_r, f_cls_r),
            "tf": (p_cls_t, r_cls_t, f_cls_t),
            "fusion_class": (p_cls_f, r_cls_f, f_cls_f)
        }
    }

def train_fold_rf(X_train, y_train):
    sm = SMOTE(random_state=SEED)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_res, y_res)
    return rf

def train_fold_tf(X_train, y_train, input_dim, cls_num):
    ros = RandomOverSampler(random_state=SEED)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X_res, y_res, test_size=0.1, random_state=SEED, stratify=y_res)
    
    train_dl = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)), batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.FloatTensor(X_vl), torch.LongTensor(y_vl)), batch_size=VAL_BATCH_SIZE)
    
    model, crit, opt, sch = init_tf(input_dim, cls_num)
    best_mcc = -1
    best_state = None
    es = 0
    
    for e in range(EPOCHS):
        train_epoch(model, train_dl, crit, opt, device)
        val_res = eval_tf(model, val_dl, crit, device)
        sch.step()
        
        if val_res[1] > best_mcc:
            best_mcc = val_res[1]
            best_state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1
            if es >= PATIENCE:
                break
    
    model.load_state_dict(best_state)
    return model

def main():
    rf_model, _, _ = train_rf()
    tf_model, tf_in_dim, tf_cls_num, _, _ = train_tf()
    
    Xr, Xt, y = load_weight_set()
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_alphas = []
    fold_class_alphas = []
    fold_macro_metrics = []
    fold_class_macro_metrics = []
    fold_cls_metrics = []
    fold_class_cls_metrics = []
    
    print("\nStarting 10-Fold CV...")
    with tqdm(total=10, desc="CV Progress", unit="fold") as pbar:
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(Xr, y)):
            Xr_train_all, Xr_test = Xr[train_idx], Xr[test_idx]
            Xt_train_all, Xt_test = Xt[train_idx], Xt[test_idx]
            y_train_all, y_test = y[train_idx], y[test_idx]
            
            Xr_train, Xr_val, y_train, y_val = train_test_split(
                Xr_train_all, y_train_all, test_size=0.1, 
                stratify=y_train_all, random_state=SEED
            )
            Xt_train, Xt_val, _, _ = train_test_split(
                Xt_train_all, y_train_all, test_size=0.1, 
                stratify=y_train_all, random_state=SEED
            )
            
            best_alpha = search_fold_weights(
                rf_model, tf_model, tf_in_dim, tf_cls_num,
                Xr_val, Xt_val, y_val
            )
            fold_alphas.append(best_alpha)
            
            best_class_alpha = search_fold_class_weights(
                rf_model, tf_model, tf_in_dim, tf_cls_num,
                Xr_val, Xt_val, y_val
            )
            fold_class_alphas.append(best_class_alpha)
            
            fold_metric = eval_fold(
                rf_model, tf_model, tf_in_dim, tf_cls_num,
                best_alpha, Xr_test, Xt_test, y_test
            )
            fold_macro_metrics.append(fold_metric["macro"])
            fold_cls_metrics.append(fold_metric["cls"])
            
            fold_class_metric = eval_fold_class_weights(
                rf_model, tf_model, tf_in_dim, tf_cls_num,
                best_class_alpha, Xr_test, Xt_test, y_test
            )
            fold_class_macro_metrics.append(fold_class_metric["macro"])
            fold_class_cls_metrics.append(fold_class_metric["cls"])
            
            pbar.update(1)
    
    avg_alpha = np.mean(fold_alphas)
    avg_class_alphas = np.mean(fold_class_alphas, axis=0)
    
    metrics_keys = ["rf", "tf", "fusion"]
    avg_macro = {}
    for key in metrics_keys:
        all_metrics = np.array([fm[key] for fm in fold_macro_metrics])
        avg_macro[key] = np.mean(all_metrics, axis=0)
    
    avg_cls = {}
    for key in metrics_keys:
        all_p = np.array([cm[key][0] for cm in fold_cls_metrics])
        all_r = np.array([cm[key][1] for cm in fold_cls_metrics])
        all_f = np.array([cm[key][2] for cm in fold_cls_metrics])
        avg_p = np.mean(all_p, axis=0)
        avg_r = np.mean(all_r, axis=0)
        avg_f = np.mean(all_f, axis=0)
        avg_cls[key] = (avg_p, avg_r, avg_f)
    
    class_metrics_keys = ["rf", "tf", "fusion_class"]
    avg_class_macro = {}
    for key in class_metrics_keys:
        all_metrics = np.array([fm[key] for fm in fold_class_macro_metrics])
        avg_class_macro[key] = np.mean(all_metrics, axis=0)
    
    avg_class_cls = {}
    for key in class_metrics_keys:
        all_p = np.array([cm[key][0] for cm in fold_class_cls_metrics])
        all_r = np.array([cm[key][1] for cm in fold_class_cls_metrics])
        all_f = np.array([cm[key][2] for cm in fold_class_cls_metrics])
        avg_p = np.mean(all_p, axis=0)
        avg_r = np.mean(all_r, axis=0)
        avg_f = np.mean(all_f, axis=0)
        avg_class_cls[key] = (avg_p, avg_r, avg_f)
    
    print("\n====================================================================")
    print("10-Fold CV Average Optimal Fusion Weights")
    print("====================================================================")
    print(f"Global RF Weight (alpha): {avg_alpha:.4f}")
    print(f"Global TF Weight (1-alpha): {1-avg_alpha:.4f}")
    print("--------------------------------------------------------------------")
    print("Per-Class RF Weights (alpha for each class):")
    for cls_idx, cls_name in LABEL_MAP.items():
        print(f"{cls_name}: {avg_class_alphas[cls_idx]:.4f} (TF weight: {1-avg_class_alphas[cls_idx]:.4f})")
    print("====================================================================\n")
    
    print("Loading weight dataset for combined CV...")
    Xr_weight, Xt_weight, y_weight = load_weight_set()
    Xr_train_base, y_train_base = load_data(TRAIN_FILES, os.path.join(FUSION_TRAIN_DIR, RF_FEATURE_FILE))
    Xt_train_base, _ = load_data(TRAIN_FILES, os.path.join(FUSION_TRAIN_DIR, TF_FEATURE_FILE))
    
    skf_weight = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    
    comb_fold_macro = []
    comb_fold_cls = []
    print("\nStarting 10-Fold CV on Combined Model (Train+Weight) with Weight Test Fold ...")
    with tqdm(total=10, desc="Combined CV Progress", unit="fold") as pbar_comb:
        for fold_idx, (train_idx, test_idx) in enumerate(skf_weight.split(Xr_weight, y_weight)):
            Xr_te = Xr_weight[test_idx]
            Xt_te = Xt_weight[test_idx]
            y_te = y_weight[test_idx]
            
            Xr_comb = np.concatenate([Xr_train_base, Xr_weight[train_idx]], axis=0)
            Xt_comb = np.concatenate([Xt_train_base, Xt_weight[train_idx]], axis=0)
            y_comb = np.concatenate([y_train_base, y_weight[train_idx]], axis=0)
            
            unique, cnt = np.unique(y_comb, return_counts=True)
            cls_num_comb = [cnt[np.where(unique==i)[0][0]] if i in unique else 0 for i in range(NUM_CLASSES)]
            input_dim_comb = Xt_comb.shape[1]
            
            rf_comb = train_fold_rf(Xr_comb, y_comb)
            tf_comb = train_fold_tf(Xt_comb, y_comb, input_dim_comb, cls_num_comb)
            
            prf_comb = rf_comb.predict_proba(Xr_te)
            pred_rf_comb = np.argmax(prf_comb, axis=1)
            acc_r_comb = accuracy_score(y_te, pred_rf_comb)
            mcc_r_comb = matthews_corrcoef(y_te, pred_rf_comb)
            bacc_r_comb = balanced_accuracy_score(y_te, pred_rf_comb)
            p_r_comb, r_r_comb, f_r_comb, _ = precision_recall_fscore_support(y_te, pred_rf_comb, average='macro', zero_division=0)
            p_cls_r_comb, r_cls_r_comb, f_cls_r_comb, _ = precision_recall_fscore_support(y_te, pred_rf_comb, average=None, zero_division=0)
            yb_comb = label_binarize(y_te, classes=range(NUM_CLASSES))
            auc_r_comb = roc_auc_score(yb_comb, prf_comb, average='macro', multi_class='ovr')
            
            te_dl = DataLoader(TensorDataset(torch.FloatTensor(Xt_te), torch.LongTensor(y_te)), batch_size=VAL_BATCH_SIZE)
            _, crit_comb, _, _ = init_tf(input_dim_comb, cls_num_comb)
            acc_t_comb, mcc_t_comb, bacc_t_comb, p_t_comb, r_t_comb, f_t_comb, p_cls_t_comb, r_cls_t_comb, f_cls_t_comb, auc_t_comb, _ = eval_tf(tf_comb, te_dl, crit_comb, device, need_probs=True)
            
            comb_fold_macro.append({
                "rf_comb": (acc_r_comb, mcc_r_comb, bacc_r_comb, p_r_comb, r_r_comb, f_r_comb, auc_r_comb),
                "tf_comb": (acc_t_comb, mcc_t_comb, bacc_t_comb, p_t_comb, r_t_comb, f_t_comb, auc_t_comb)
            })
            comb_fold_cls.append({
                "rf_comb": (p_cls_r_comb, r_cls_r_comb, f_cls_r_comb),
                "tf_comb": (p_cls_t_comb, r_cls_t_comb, f_cls_t_comb)
            })
            
            pbar_comb.update(1)
    
    avg_comb_macro = {}
    for key in ["rf_comb", "tf_comb"]:
        all_metrics = np.array([fm[key] for fm in comb_fold_macro])
        avg_comb_macro[key] = np.mean(all_metrics, axis=0)
    
    avg_comb_cls = {}
    for key in ["rf_comb", "tf_comb"]:
        all_p = np.array([cm[key][0] for cm in comb_fold_cls])
        all_r = np.array([cm[key][1] for cm in comb_fold_cls])
        all_f = np.array([cm[key][2] for cm in comb_fold_cls])
        avg_p = np.mean(all_p, axis=0)
        avg_r = np.mean(all_r, axis=0)
        avg_f = np.mean(all_f, axis=0)
        avg_comb_cls[key] = (avg_p, avg_r, avg_f)
    
    print("==============================================================================================================")
    print("10-Fold CV Average Macro Results (Including Combined Train+Weight Dataset and Per-Class Fusion)")
    print("==============================================================================================================")
    print(f"{'Model':<30} {'ACC':<10} {'MCC':<10} {'BACC':<10} {'Precision':<12} {'Recall':<10} {'Fscore':<10} {'AUC':<10}")
    print("--------------------------------------------------------------------------------------------------------------")
    rf_m = avg_macro["rf"]
    print(f"{'RandomForest (Train)':<30} {rf_m[0]:<10.4f} {rf_m[1]:<10.4f} {rf_m[2]:<10.4f} {rf_m[3]:<12.4f} {rf_m[4]:<10.4f} {rf_m[5]:<10.4f} {rf_m[6]:<10.4f}")
    rf_comb_m = avg_comb_macro["rf_comb"]
    print(f"{'RandomForest (Combined)':<30} {rf_comb_m[0]:<10.4f} {rf_comb_m[1]:<10.4f} {rf_comb_m[2]:<10.4f} {rf_comb_m[3]:<12.4f} {rf_comb_m[4]:<10.4f} {rf_comb_m[5]:<10.4f} {rf_comb_m[6]:<10.4f}")
    tf_m = avg_macro["tf"]
    print(f"{'Transformer (Train)':<30} {tf_m[0]:<10.4f} {tf_m[1]:<10.4f} {tf_m[2]:<10.4f} {tf_m[3]:<12.4f} {tf_m[4]:<10.4f} {tf_m[5]:<10.4f} {tf_m[6]:<10.4f}")
    tf_comb_m = avg_comb_macro["tf_comb"]
    print(f"{'Transformer (Combined)':<30} {tf_comb_m[0]:<10.4f} {tf_comb_m[1]:<10.4f} {tf_comb_m[2]:<10.4f} {tf_comb_m[3]:<12.4f} {tf_comb_m[4]:<10.4f} {tf_comb_m[5]:<10.4f} {tf_comb_m[6]:<10.4f}")
    fusion_m = avg_macro["fusion"]
    print(f"{'Fusion (Global Weight)':<30} {fusion_m[0]:<10.4f} {fusion_m[1]:<10.4f} {fusion_m[2]:<10.4f} {fusion_m[3]:<12.4f} {fusion_m[4]:<10.4f} {fusion_m[5]:<10.4f} {fusion_m[6]:<10.4f}")
    fusion_class_m = avg_class_macro["fusion_class"]
    print(f"{'Fusion (Per-Class Weight)':<30} {fusion_class_m[0]:<10.4f} {fusion_class_m[1]:<10.4f} {fusion_class_m[2]:<10.4f} {fusion_class_m[3]:<12.4f} {fusion_class_m[4]:<10.4f} {fusion_class_m[5]:<10.4f} {fusion_class_m[6]:<10.4f}")
    print("==============================================================================================================\n")
    
    print("=====================================================================================================")
    print("10-Fold CV Average Per-Class Results (Precision / Recall / Fscore)")
    print("=====================================================================================================")
    print(f"{'Class':<25} {'Metric':<10} {'RF (Train)':<12} {'RF (Combined)':<12} {'TF (Train)':<12} {'TF (Combined)':<12} {'Fusion (Global)':<12} {'Fusion (Per-Class)':<12}")
    print("-----------------------------------------------------------------------------------------------------")
    for cls_idx, cls_name in LABEL_MAP.items():
        rf_p, rf_r, rf_f = avg_cls["rf"][0][cls_idx], avg_cls["rf"][1][cls_idx], avg_cls["rf"][2][cls_idx]
        rf_comb_p, rf_comb_r, rf_comb_f = avg_comb_cls["rf_comb"][0][cls_idx], avg_comb_cls["rf_comb"][1][cls_idx], avg_comb_cls["rf_comb"][2][cls_idx]
        tf_p, tf_r, tf_f = avg_cls["tf"][0][cls_idx], avg_cls["tf"][1][cls_idx], avg_cls["tf"][2][cls_idx]
        tf_comb_p, tf_comb_r, tf_comb_f = avg_comb_cls["tf_comb"][0][cls_idx], avg_comb_cls["tf_comb"][1][cls_idx], avg_comb_cls["tf_comb"][2][cls_idx]
        fusion_p, fusion_r, fusion_f = avg_cls["fusion"][0][cls_idx], avg_cls["fusion"][1][cls_idx], avg_cls["fusion"][2][cls_idx]
        fusion_class_p, fusion_class_r, fusion_class_f = avg_class_cls["fusion_class"][0][cls_idx], avg_class_cls["fusion_class"][1][cls_idx], avg_class_cls["fusion_class"][2][cls_idx]
        
        print(f"{cls_name:<25} {'Precision':<10} {rf_p:<12.4f} {rf_comb_p:<12.4f} {tf_p:<12.4f} {tf_comb_p:<12.4f} {fusion_p:<12.4f} {fusion_class_p:<12.4f}")
        print(f"{cls_name:<25} {'Recall':<10} {rf_r:<12.4f} {rf_comb_r:<12.4f} {tf_r:<12.4f} {tf_comb_r:<12.4f} {fusion_r:<12.4f} {fusion_class_r:<12.4f}")
        print(f"{cls_name:<25} {'Fscore':<10} {rf_f:<12.4f} {rf_comb_f:<12.4f} {tf_f:<12.4f} {tf_comb_f:<12.4f} {fusion_f:<12.4f} {fusion_class_f:<12.4f}")
        print("-----------------------------------------------------------------------------------------------------")
    
    print("\nAll 10-Fold CV evaluation finished!")

if __name__ == "__main__":
    main()