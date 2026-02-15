import warnings
import random
import os
import copy
import math
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
from itertools import product
from bayes_opt import BayesianOptimization, UtilityFunction

warnings.filterwarnings("ignore")

SEED = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(False)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
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

BATCH_SIZE = 512
VAL_BATCH_SIZE = 2048
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 4
SEARCH_PATIENCE = 6
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

RF_PARAMS_MAP = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 8, 10],
    'min_samples_leaf': [1, 2, 5, 8],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [False, True]
}

TF_PARAMS_MAP = {
    'd_model': [16, 32, 64],
    'nhead': [2, 4],
    'num_layers': [1, 2, 3],
    'dim_feedforward': [32, 64, 128],
    'dropout': [0.0, 0.1, 0.2]
}

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
    
    if return_cls_num:
        unique, cnt = np.unique(labels, return_counts=True)
        cls_num = [cnt[np.where(unique==i)[0][0]] if i in unique else 0 for i in range(NUM_CLASSES)]
        return feat, labels, cls_num
    return feat, labels

class DNA_Transformer_Model(nn.Module):
    def __init__(self, input_dim=16, num_classes=5, d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model, bias=False)
        self.relu = nn.ReLU()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
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

def init_tf(input_dim, cls_num, d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.1):
    model = DNA_Transformer_Model(input_dim=input_dim, num_classes=NUM_CLASSES, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout).to(device)
    weights = 1.0 / np.array(cls_num)
    weights = weights / weights.sum() * NUM_CLASSES
    weights = torch.from_numpy(weights).float().to(device)
    crit = LDAMLoss(cls_num_list=cls_num, weight=weights)
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sch = StepLR(opt, step_size=10, gamma=0.5)
    return model, crit, opt, sch

def joint_evaluate(rf_params, tf_params, X_rf, X_tf, y, cls_num):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    fold_fusion_mcc = []
    
    for train_idx, val_idx in skf.split(X_rf, y):
        Xr_train, Xr_val = X_rf[train_idx], X_rf[val_idx]
        Xt_train, Xt_val = X_tf[train_idx], X_tf[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        sm = SMOTE(random_state=SEED)
        Xr_res, yr_res = sm.fit_resample(Xr_train, y_train)
        ros = RandomOverSampler(random_state=SEED)
        Xt_res, yt_res = ros.fit_resample(Xt_train, y_train)
        
        rf = RandomForestClassifier(**rf_params, random_state=SEED, n_jobs=-1)
        rf.fit(Xr_res, yr_res)
        prf = rf.predict_proba(Xr_val)
        
        Xt_tr, Xt_v, yt_tr, yt_v = train_test_split(Xt_res, yt_res, test_size=0.1, random_state=SEED, stratify=yt_res)
        tl = DataLoader(TensorDataset(torch.FloatTensor(Xt_tr), torch.LongTensor(yt_tr)), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        vl = DataLoader(TensorDataset(torch.FloatTensor(Xt_v), torch.LongTensor(yt_v)), batch_size=VAL_BATCH_SIZE, pin_memory=True)
        
        model, crit, opt, sch = init_tf(X_tf.shape[1], cls_num, **tf_params)
        best_mcc = -1
        es = 0
        for e in range(EPOCHS):
            train_epoch(model, tl, crit, opt, device)
            res = eval_tf(model, vl, crit, device)
            sch.step()
            if res[1] > best_mcc:
                best_mcc = res[1]
                es = 0
            else:
                es += 1
                if es >= SEARCH_PATIENCE:
                    break
        
        test_loader = DataLoader(TensorDataset(torch.FloatTensor(Xt_val), torch.LongTensor(y_val)), batch_size=VAL_BATCH_SIZE, pin_memory=True)
        ptf = eval_tf(model, test_loader, crit, device, need_probs=True)[-1]
        
        fused_probs = 0.5 * prf + 0.5 * ptf
        pred_f = np.argmax(fused_probs, axis=1)
        fusion_mcc = matthews_corrcoef(y_val, pred_f)
        fold_fusion_mcc.append(fusion_mcc)
    
    return np.mean(fold_fusion_mcc)

def bo_joint_search(X_rf, X_tf, y, cls_num):
    print("Bayesian Optimization for Joint RF-TF Hyperparameters (Fusion-Oriented)...")
    
    def target_function(
        rf_n_estimators_idx, rf_max_depth_idx, rf_min_samples_split_idx,
        rf_min_samples_leaf_idx, rf_max_features_idx, rf_bootstrap_idx,
        tf_d_model_idx, tf_nhead_idx, tf_num_layers_idx,
        tf_dim_feedforward_idx, tf_dropout_idx
    ):
        rf_params = {
            'n_estimators': RF_PARAMS_MAP['n_estimators'][int(round(rf_n_estimators_idx))],
            'max_depth': RF_PARAMS_MAP['max_depth'][int(round(rf_max_depth_idx))],
            'min_samples_split': RF_PARAMS_MAP['min_samples_split'][int(round(rf_min_samples_split_idx))],
            'min_samples_leaf': RF_PARAMS_MAP['min_samples_leaf'][int(round(rf_min_samples_leaf_idx))],
            'max_features': 'sqrt' if int(round(rf_max_features_idx)) == 0 else 'log2',
            'bootstrap': bool(int(round(rf_bootstrap_idx)))
        }
        
        if rf_params['max_depth'] == 0:
            rf_params['max_depth'] = None
        
        tf_params = {
            'd_model': TF_PARAMS_MAP['d_model'][int(round(tf_d_model_idx))],
            'nhead': TF_PARAMS_MAP['nhead'][int(round(tf_nhead_idx))],
            'num_layers': TF_PARAMS_MAP['num_layers'][int(round(tf_num_layers_idx))],
            'dim_feedforward': TF_PARAMS_MAP['dim_feedforward'][int(round(tf_dim_feedforward_idx))],
            'dropout': TF_PARAMS_MAP['dropout'][int(round(tf_dropout_idx))]
        }
        
        if tf_params['d_model'] % tf_params['nhead'] != 0:
            return -1.0
        
        score = joint_evaluate(rf_params, tf_params, X_rf, X_tf, y, cls_num)
        return score
    
    pbounds = {
        'rf_n_estimators_idx': (0, len(RF_PARAMS_MAP['n_estimators'])-1),
        'rf_max_depth_idx': (0, len(RF_PARAMS_MAP['max_depth'])-1),
        'rf_min_samples_split_idx': (0, len(RF_PARAMS_MAP['min_samples_split'])-1),
        'rf_min_samples_leaf_idx': (0, len(RF_PARAMS_MAP['min_samples_leaf'])-1),
        'rf_max_features_idx': (0, len(RF_PARAMS_MAP['max_features'])-1),
        'rf_bootstrap_idx': (0, len(RF_PARAMS_MAP['bootstrap'])-1),
        'tf_d_model_idx': (0, len(TF_PARAMS_MAP['d_model'])-1),
        'tf_nhead_idx': (0, len(TF_PARAMS_MAP['nhead'])-1),
        'tf_num_layers_idx': (0, len(TF_PARAMS_MAP['num_layers'])-1),
        'tf_dim_feedforward_idx': (0, len(TF_PARAMS_MAP['dim_feedforward'])-1),
        'tf_dropout_idx': (0, len(TF_PARAMS_MAP['dropout'])-1)
    }
    
    optimizer = BayesianOptimization(
        f=target_function,
        pbounds=pbounds,
        random_state=SEED,
        verbose=2
    )
    
    utility = UtilityFunction(kind="ei")
    
    optimizer.maximize(
        init_points=10,
        n_iter=40,
        acquisition_function=utility
    )
    
    best_params = optimizer.max['params']
    best_rf_params = {
        'n_estimators': RF_PARAMS_MAP['n_estimators'][int(round(best_params['rf_n_estimators_idx']))],
        'max_depth': RF_PARAMS_MAP['max_depth'][int(round(best_params['rf_max_depth_idx']))],
        'min_samples_split': RF_PARAMS_MAP['min_samples_split'][int(round(best_params['rf_min_samples_split_idx']))],
        'min_samples_leaf': RF_PARAMS_MAP['min_samples_leaf'][int(round(best_params['rf_min_samples_leaf_idx']))],
        'max_features': 'sqrt' if int(round(best_params['rf_max_features_idx'])) == 0 else 'log2',
        'bootstrap': bool(int(round(best_params['rf_bootstrap_idx'])))
    }
    if best_rf_params['max_depth'] == 0:
        best_rf_params['max_depth'] = None
    
    best_tf_params = {
        'd_model': TF_PARAMS_MAP['d_model'][int(round(best_params['tf_d_model_idx']))],
        'nhead': TF_PARAMS_MAP['nhead'][int(round(best_params['tf_nhead_idx']))],
        'num_layers': TF_PARAMS_MAP['num_layers'][int(round(best_params['tf_num_layers_idx']))],
        'dim_feedforward': TF_PARAMS_MAP['dim_feedforward'][int(round(best_params['tf_dim_feedforward_idx']))],
        'dropout': TF_PARAMS_MAP['dropout'][int(round(best_params['tf_dropout_idx']))]
    }
    
    print(f"\nBest Fusion MCC: {optimizer.max['target']:.4f}")
    return best_rf_params, best_tf_params

def train_rf(best_params):
    path = os.path.join(FUSION_TRAIN_DIR, RF_FEATURE_FILE)
    X, y = load_data(TRAIN_FILES, path)
    sm = SMOTE(random_state=SEED)
    Xr, yr = sm.fit_resample(X, y)
    rf_model = RandomForestClassifier(**best_params, random_state=SEED, n_jobs=-1)
    rf_model.fit(Xr, yr)
    return rf_model, X, y

def train_tf(best_params):
    path = os.path.join(FUSION_TRAIN_DIR, TF_FEATURE_FILE)
    X, y, cls_num = load_data(TRAIN_FILES, path, return_cls_num=True)
    ros = RandomOverSampler(random_state=SEED)
    Xr, yr = ros.fit_resample(X, y)
    Xt, Xv, yt, yv = train_test_split(Xr, yr, test_size=0.1, random_state=SEED, stratify=yr)
    
    tl = DataLoader(TensorDataset(torch.FloatTensor(Xt), torch.LongTensor(yt)), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    vl = DataLoader(TensorDataset(torch.FloatTensor(Xv), torch.LongTensor(yv)), batch_size=VAL_BATCH_SIZE, pin_memory=True)
    
    model, crit, opt, sch = init_tf(X.shape[1], cls_num, **best_params)
    best_mcc, best_state, es = -1, None, 0
    
    for e in range(EPOCHS):
        train_epoch(model, tl, crit, opt, device)
        res = eval_tf(model, vl, crit, device)
        sch.step()
        if res[1] > best_mcc:
            best_mcc = res[1]
            best_state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1
            if es >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, X.shape[1], cls_num, X, y

def load_weight_set():
    path_rf = os.path.join(FUSION_WEIGHT_DIR, RF_FEATURE_FILE)
    path_tf = os.path.join(FUSION_WEIGHT_DIR, TF_FEATURE_FILE)
    Xr, y = load_data(WEIGHT_FILES, path_rf)
    Xt, _ = load_data(WEIGHT_FILES, path_tf)
    return Xr, Xt, y

def search_fold_class_weights(rf, tf, input_dim, cls_num, Xr_val, Xt_val, y_val):
    tl = DataLoader(TensorDataset(torch.FloatTensor(Xt_val), torch.LongTensor(y_val)), batch_size=VAL_BATCH_SIZE)
    _, crit, _, _ = init_tf(input_dim, cls_num)
    
    prf = rf.predict_proba(Xr_val)
    ptf = eval_tf(tf, tl, crit, device, need_probs=True)[-1]
    
    best_mcc = -1
    best_alpha = np.array([0.5]*NUM_CLASSES)
    alphas = np.linspace(0.0, 1.0, 11)
    combinations = list(product(alphas, repeat=NUM_CLASSES))
    
    with tqdm(total=len(combinations), desc="Class Weight Search") as pbar:
        for combo in combinations:
            class_alphas = np.array(combo)
            fused_probs = class_alphas * prf + (1 - class_alphas) * ptf
            pred = np.argmax(fused_probs, axis=1)
            mcc = matthews_corrcoef(y_val, pred)
            
            if mcc > best_mcc:
                best_mcc = mcc
                best_alpha = class_alphas
            pbar.update(1)
    return best_alpha

def eval_fold_class_weights(rf, tf, input_dim, cls_num, class_alphas, Xr_test, Xt_test, y_test):
    prf = rf.predict_proba(Xr_test)
    tl = DataLoader(TensorDataset(torch.FloatTensor(Xt_test), torch.LongTensor(y_test)), batch_size=VAL_BATCH_SIZE)
    _, crit, _, _ = init_tf(input_dim, cls_num)
    *_, ptf = eval_tf(tf, tl, crit, device, need_probs=True)
    
    fused_probs = class_alphas * prf + (1 - class_alphas) * ptf
    pred_f = np.argmax(fused_probs, axis=1)
    
    acc_f = accuracy_score(y_test, pred_f)
    mcc_f = matthews_corrcoef(y_test, pred_f)
    bacc_f = balanced_accuracy_score(y_test, pred_f)
    p_f, r_f, f_f, _ = precision_recall_fscore_support(y_test, pred_f, average='macro', zero_division=0)
    p_cls_f, r_cls_f, f_cls_f, _ = precision_recall_fscore_support(y_test, pred_f, average=None, zero_division=0)
    yb = label_binarize(y_test, classes=range(NUM_CLASSES))
    auc_f = roc_auc_score(yb, fused_probs, average='macro', multi_class='ovr')
    
    return {
        "macro": (acc_f, mcc_f, bacc_f, p_f, r_f, f_f, auc_f),
        "cls": (p_cls_f, r_cls_f, f_cls_f)
    }

def main():
    rf_feature_path = os.path.join(FUSION_TRAIN_DIR, RF_FEATURE_FILE)
    tf_feature_path = os.path.join(FUSION_TRAIN_DIR, TF_FEATURE_FILE)
    
    X_rf, y_rf = load_data(TRAIN_FILES, rf_feature_path)
    X_tf, y_tf, cls_num_tf = load_data(TRAIN_FILES, tf_feature_path, return_cls_num=True)
    
    best_rf_params, best_tf_params = bo_joint_search(X_rf, X_tf, y_rf, cls_num_tf)
    
    print(f"\nOptimal RandomForest Parameters: {best_rf_params}")
    print(f"Optimal Transformer Parameters: {best_tf_params}")
    
    rf_model, _, _ = train_rf(best_rf_params)
    tf_model, tf_in_dim, tf_cls_num, _, _ = train_tf(best_tf_params)
    
    Xr, Xt, y = load_weight_set()
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_class_alphas = []
    fold_fusion_macro = []
    fold_fusion_cls = []
    
    print("\nStarting 10-Fold CV for Per-Class Weight Fusion...")
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
            
            best_class_alpha = search_fold_class_weights(
                rf_model, tf_model, tf_in_dim, tf_cls_num,
                Xr_val, Xt_val, y_val
            )
            fold_class_alphas.append(best_class_alpha)
            
            fold_fusion_metric = eval_fold_class_weights(
                rf_model, tf_model, tf_in_dim, tf_cls_num,
                best_class_alpha, Xr_test, Xt_test, y_test
            )
            fold_fusion_macro.append(fold_fusion_metric["macro"])
            fold_fusion_cls.append(fold_fusion_metric["cls"])
            
            pbar.update(1)
    
    avg_class_alphas = np.mean(fold_class_alphas, axis=0)
    avg_fusion_macro = np.mean(fold_fusion_macro, axis=0)
    
    all_p = np.array([cm[0] for cm in fold_fusion_cls])
    all_r = np.array([cm[1] for cm in fold_fusion_cls])
    all_f = np.array([cm[2] for cm in fold_fusion_cls])
    avg_p = np.mean(all_p, axis=0)
    avg_r = np.mean(all_r, axis=0)
    avg_f = np.mean(all_f, axis=0)
    
    print("\n====================================================================")
    print("10-Fold CV Average Optimal Per-Class Fusion Weights")
    print("====================================================================")
    for cls_idx, cls_name in LABEL_MAP.items():
        print(f"{cls_name}: {avg_class_alphas[cls_idx]:.4f} (TF weight: {1-avg_class_alphas[cls_idx]:.4f})")
    print("====================================================================\n")
    
    print("==============================================================================================================")
    print("10-Fold CV Average Results (Fusion with Per-Class Weights)")
    print("==============================================================================================================")
    print(f"{'Metric':<12} {'Value':<10}")
    print("--------------------------------------------------------------------------------------------------------------")
    print(f"{'ACC':<12} {avg_fusion_macro[0]:<10.4f}")
    print(f"{'MCC':<12} {avg_fusion_macro[1]:<10.4f}")
    print(f"{'BACC':<12} {avg_fusion_macro[2]:<10.4f}")
    print(f"{'Precision':<12} {avg_fusion_macro[3]:<10.4f}")
    print(f"{'Recall':<12} {avg_fusion_macro[4]:<10.4f}")
    print(f"{'Fscore':<12} {avg_fusion_macro[5]:<10.4f}")
    print(f"{'AUC':<12} {avg_fusion_macro[6]:<10.4f}")
    print("==============================================================================================================\n")
    
    print("=====================================================================================================")
    print("10-Fold CV Average Per-Class Results (Precision / Recall / Fscore)")
    print("=====================================================================================================")
    print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'Fscore':<12}")
    print("-----------------------------------------------------------------------------------------------------")
    for cls_idx, cls_name in LABEL_MAP.items():
        print(f"{cls_name:<25} {avg_p[cls_idx]:<12.4f} {avg_r[cls_idx]:<12.4f} {avg_f[cls_idx]:<12.4f}")
    print("-----------------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()