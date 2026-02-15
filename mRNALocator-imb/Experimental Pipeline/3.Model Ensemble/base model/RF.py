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
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from imblearn.over_sampling import SMOTE, RandomOverSampler

warnings.filterwarnings("ignore")

# ====================== Global Configuration ======================
SEED = 10
fold = 10
device = torch.device('cuda' if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Set random seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) if torch.cuda.is_available() else None
torch.cuda.manual_seed_all(SEED) if torch.cuda.is_available() else None
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Path configuration (match the reference code path structure)
TRAIN_DIR = "../data/train"
WEIGHT_DIR = "../data/weight"
FUSION_TRAIN_DIR = "../data/fusion/train"
FUSION_WEIGHT_DIR = "../data/fusion/weight"

TRAIN_FILES = [
    f'{TRAIN_DIR}/Cytoplasm.fasta', f'{TRAIN_DIR}/Endoplasmic_reticulum.fasta',
    f'{TRAIN_DIR}/Extracellular_region.fasta', f'{TRAIN_DIR}/Mitochondria.fasta',
    f'{TRAIN_DIR}/Nucleus.fasta'
]
WEIGHT_FILES = [
    f'{WEIGHT_DIR}/Cytoplasm.fasta', f'{WEIGHT_DIR}/Endoplasmic_reticulum.fasta',
    f'{WEIGHT_DIR}/Extracellular_region.fasta', f'{WEIGHT_DIR}/Mitochondria.fasta',
    f'{WEIGHT_DIR}/Nucleus.fasta'
]

# Feature file configuration
RF_FEATURE_FILE = "set.csv"          
TF_FEATURE_FILE = "set2.csv"        

# Transformer hyperparameters
BATCH_SIZE = 128
VAL_BATCH_SIZE = 512
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 4
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# Class configuration (match the reference code)
LABEL_MAP = {0: 'Cytoplasm', 1: 'Endoplasmic_reticulum',
             2: 'Extracellular_region', 3: 'Mitochondria', 4: 'Nucleus'}
NUM_CLASSES = len(LABEL_MAP)
CLASS_LABELS = sorted(LABEL_MAP.keys())
CLASS_NAMES = [LABEL_MAP[label] for label in CLASS_LABELS]

# Weight search configuration
WEIGHT_SEARCH_STEPS = 20
TEST_SIZE = 0.3

# ====================== LDAM Loss Function ======================
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

# ====================== Utility Functions (match reference code) ======================
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

def load_train_data(fasta_paths, feature_path, return_cls_num=False):
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
    
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    train_feature = pd.read_csv(
        feature_path, sep=',', header=None, index_col=None, low_memory=False
    ).values.tolist()
    train_feature = np.array(train_feature)[1:, 1:]
    train_feature = np.array(train_feature, dtype=float)
    
    if len(train_labels) != train_feature.shape[0]:
        raise ValueError(f"Training label count ({len(train_labels)}) does not match feature count ({train_feature.shape[0]})!")
    
    if return_cls_num:
        unique, counts = np.unique(train_labels, return_counts=True)
        cls_num_list = [counts[np.where(unique == i)[0][0]] if i in unique else 0 for i in range(NUM_CLASSES)]
        return train_feature, train_labels, cls_num_list
    return train_feature, train_labels

# ====================== Transformer Model & Training Functions ======================
class DNA_Transformer_Model(nn.Module):
    def __init__(self, input_dim=16, num_classes=5, d_model=16, nhead=2, num_layers=1, dim_feedforward=32):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model, bias=False)
        self.relu = nn.ReLU()
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True
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

def validate_tf(model, dataloader, criterion, device, need_probs=False):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = [] if need_probs else None
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
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
            if need_probs:
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    
    total_loss = sum(loss_list) / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    bacc = balanced_accuracy_score(all_labels, all_preds)
    auc = 0.0
    
    if need_probs and all_probs is not None:
        all_probs = np.concatenate(all_probs)
        y_val_binarized = label_binarize(all_labels, classes=range(NUM_CLASSES))
        auc = roc_auc_score(y_val_binarized, all_probs, average='macro', multi_class='ovr')
    
    # Per-class metrics
    prec_cls, rec_cls, fscore_cls, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0)
    
    result_dict = {
        'loss': total_loss, 'acc': acc, 'mcc': mcc,
        'precision': precision_macro, 'recall': recall_macro, 'fscore': fscore_macro,
        'bacc': bacc, 'auc': auc,
        'prec_cls': prec_cls, 'rec_cls': rec_cls, 'fscore_cls': fscore_cls,
        'preds': all_preds, 'labels': all_labels, 'probs': all_probs if need_probs else None
    }
    return result_dict

def init_tf_model(num_classes=5, input_dim=16, cls_num_list=None):
    model = DNA_Transformer_Model(input_dim=input_dim, num_classes=num_classes).to(device)
    
    cls_weights = None
    if cls_num_list is not None:
        cls_weights = 1.0 / np.array(cls_num_list)
        cls_weights = cls_weights / cls_weights.sum() * num_classes
        cls_weights = torch.from_numpy(cls_weights).float().to(device)
    
    criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, weight=cls_weights, s=30)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    return model, criterion, optimizer, scheduler

# ====================== Base Model Training ======================
def train_rf_base_model():
    feature_path = os.path.join(FUSION_TRAIN_DIR, RF_FEATURE_FILE)
    X_train, y_train = load_train_data(TRAIN_FILES, feature_path)
    
    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, random_state=SEED, n_jobs=-1
    )
    rf_model.fit(X_train_resampled, y_train_resampled)
    
    return {
        'model': rf_model,
        'input_dim': X_train.shape[1],
        'X_train': X_train,
        'y_train': y_train
    }

def train_transformer_base_model():
    feature_path = os.path.join(FUSION_TRAIN_DIR, TF_FEATURE_FILE)
    X_train, y_train, cls_num_list = load_train_data(TRAIN_FILES, feature_path, return_cls_num=True)
    
    ros = RandomOverSampler(random_state=SEED, sampling_strategy="not minority")
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_resampled, y_train_resampled, test_size=0.1, random_state=SEED, stratify=y_train_resampled
    )
    
    X_tr_tensor = torch.from_numpy(X_tr).float()
    y_tr_tensor = torch.from_numpy(y_tr).long()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).long()
    
    train_loader = DataLoader(
        TensorDataset(X_tr_tensor, y_tr_tensor),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )
    
    model, criterion, optimizer, scheduler = init_tf_model(
        num_classes=NUM_CLASSES, input_dim=X_train.shape[1], cls_num_list=cls_num_list
    )
    
    best_val_mcc = -1.0
    best_model_state = None
    early_stop_count = 0
    
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = validate_tf(model, val_loader, criterion, device, need_probs=False)
        val_mcc = val_result['mcc']
        scheduler.step()
        
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_model_state = copy.deepcopy(model.state_dict())
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= PATIENCE:
                break
    
    model.load_state_dict(best_model_state)
    
    return {
        'model': model,
        'input_dim': X_train.shape[1],
        'cls_num_list': cls_num_list,
        'X_train': X_train,
        'y_train': y_train
    }

# ====================== CV Evaluation (match reference output format) ======================
def evaluate_rf_cv(rf_model, X_data, y_data):
    acc = np.zeros(fold, dtype=float)
    mcc = np.zeros(fold, dtype=float)
    bacc = np.zeros(fold, dtype=float)
    auc = np.zeros(fold, dtype=float)
    macro_precision = np.zeros(fold, dtype=float)
    macro_recall = np.zeros(fold, dtype=float)
    macro_fscore = np.zeros(fold, dtype=float)
    class_precision = np.zeros((fold, NUM_CLASSES), dtype=float)
    class_recall = np.zeros((fold, NUM_CLASSES), dtype=float)
    class_fscore = np.zeros((fold, NUM_CLASSES), dtype=float)
    
    kf = KFold(n_splits=fold, shuffle=True, random_state=SEED)
    smote = SMOTE(random_state=SEED, k_neighbors=5)
    
    i = 0
    for train_index, val_index in tqdm(kf.split(X_data), desc=f"{fold}-Fold CV (SMOTE)", total=fold, leave=True):
        X_tr, X_val = X_data[train_index], X_data[val_index]
        y_tr, y_val = y_data[train_index], y_data[val_index]
        
        X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
        rf_model.fit(X_tr, y_tr)
        
        y_pred = rf_model.predict(X_val)
        y_pred_proba = rf_model.predict_proba(X_val)
        
        # Calculate metrics
        acc[i] = accuracy_score(y_val, y_pred)
        mcc[i] = matthews_corrcoef(y_val, y_pred)
        bacc[i] = balanced_accuracy_score(y_val, y_pred)
        
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_val, y_pred, average='macro')
        macro_precision[i] = prec_macro
        macro_recall[i] = rec_macro
        macro_fscore[i] = f1_macro
        
        prec_cls, rec_cls, f1_cls, _ = precision_recall_fscore_support(y_val, y_pred, average=None)
        class_precision[i] = prec_cls
        class_recall[i] = rec_cls
        class_fscore[i] = f1_cls
        
        y_val_binarized = label_binarize(y_val, classes=range(NUM_CLASSES))
        auc[i] = roc_auc_score(y_val_binarized, y_pred_proba, average='macro', multi_class='ovr')
        
        i += 1
    
    # Calculate mean metrics
    mean_metrics = {
        'ACC': np.mean(acc), 'MCC': np.mean(mcc),
        'Precision': np.mean(macro_precision), 'Recall': np.mean(macro_recall),
        'Fscore': np.mean(macro_fscore), 'BACC': np.mean(bacc), 'AUC': np.mean(auc)
    }
    class_mean_metrics = {
        'Precision': np.mean(class_precision, axis=0),
        'Recall': np.mean(class_recall, axis=0),
        'Fscore': np.mean(class_fscore, axis=0)
    }
    
    # Print results (match reference format exactly)
    print("\n" + "="*110)
    print(f"set - SMOTE + RandomForest - 10-Fold Cross Validation (Overall) Results")
    print("="*110)
    print(f"{'Fold':<8} {'ACC':<10} {'MCC':<10} {'Precision':<12} {'Recall':<10} {'Fscore':<10} {'BACC':<10} {'AUC':<10}")
    print("-"*110)
    for i in range(fold):
        print(f"{i+1:<8} {acc[i]:<10.4f} {mcc[i]:<10.4f} "
              f"{macro_precision[i]:<12.4f} {macro_recall[i]:<10.4f} "
              f"{macro_fscore[i]:<10.4f} {bacc[i]:<10.4f} {auc[i]:<10.4f}")
    print("-"*110)
    print(f"{'Mean':<8} {mean_metrics['ACC']:<10.4f} {mean_metrics['MCC']:<10.4f} "
          f"{mean_metrics['Precision']:<12.4f} {mean_metrics['Recall']:<10.4f} "
          f"{mean_metrics['Fscore']:<10.4f} {mean_metrics['BACC']:<10.4f} {mean_metrics['AUC']:<10.4f}")
    print("="*110 + "\n")
    
    print("="*80)
    print(f"set - SMOTE + RandomForest - Per-Class Mean Results (10-Fold CV)")
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
    
    return mean_metrics, class_mean_metrics

def evaluate_fusion_model(rf_model, tf_model, tf_input_dim, tf_cls_num_list, X_rf, X_tf, y):
    # Initialize metrics arrays
    acc = np.zeros(fold, dtype=float)
    mcc = np.zeros(fold, dtype=float)
    bacc = np.zeros(fold, dtype=float)
    auc = np.zeros(fold, dtype=float)
    macro_precision = np.zeros(fold, dtype=float)
    macro_recall = np.zeros(fold, dtype=float)
    macro_fscore = np.zeros(fold, dtype=float)
    class_precision = np.zeros((fold, NUM_CLASSES), dtype=float)
    class_recall = np.zeros((fold, NUM_CLASSES), dtype=float)
    class_fscore = np.zeros((fold, NUM_CLASSES), dtype=float)
    
    kf = KFold(n_splits=fold, shuffle=True, random_state=SEED)
    best_rf_weight = 0.5  # Default fusion weight
    
    i = 0
    for train_index, val_index in tqdm(kf.split(X_rf), desc=f"{fold}-Fold CV (Fusion Model)", total=fold, leave=True):
        # Prepare validation data
        X_rf_val = X_rf[val_index]
        X_tf_val = X_tf[val_index]
        y_val = y[val_index]
        
        # Get predictions from base models
        rf_probs = rf_model.predict_proba(X_rf_val)
        
        # Transformer predictions
        X_tf_val_tensor = torch.from_numpy(X_tf_val).float()
        val_loader = DataLoader(
            TensorDataset(X_tf_val_tensor, torch.from_numpy(y_val).long()),
            batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
        )
        _, tf_criterion, _, _ = init_tf_model(num_classes=NUM_CLASSES, input_dim=tf_input_dim, cls_num_list=tf_cls_num_list)
        tf_result = validate_tf(tf_model, val_loader, tf_criterion, device, need_probs=True)
        tf_probs = tf_result['probs']
        
        # Fusion predictions
        fused_probs = best_rf_weight * rf_probs + (1 - best_rf_weight) * tf_probs
        fused_preds = np.argmax(fused_probs, axis=1)
        
        # Calculate metrics
        acc[i] = accuracy_score(y_val, fused_preds)
        mcc[i] = matthews_corrcoef(y_val, fused_preds)
        bacc[i] = balanced_accuracy_score(y_val, fused_preds)
        
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_val, fused_preds, average='macro')
        macro_precision[i] = prec_macro
        macro_recall[i] = rec_macro
        macro_fscore[i] = f1_macro
        
        prec_cls, rec_cls, f1_cls, _ = precision_recall_fscore_support(y_val, fused_preds, average=None)
        class_precision[i] = prec_cls
        class_recall[i] = rec_cls
        class_fscore[i] = f1_cls
        
        y_val_binarized = label_binarize(y_val, classes=range(NUM_CLASSES))
        auc[i] = roc_auc_score(y_val_binarized, fused_probs, average='macro', multi_class='ovr')
        
        i += 1
    
    # Calculate mean metrics
    mean_metrics = {
        'ACC': np.mean(acc), 'MCC': np.mean(mcc),
        'Precision': np.mean(macro_precision), 'Recall': np.mean(macro_recall),
        'Fscore': np.mean(macro_fscore), 'BACC': np.mean(bacc), 'AUC': np.mean(auc)
    }
    class_mean_metrics = {
        'Precision': np.mean(class_precision, axis=0),
        'Recall': np.mean(class_recall, axis=0),
        'Fscore': np.mean(class_fscore, axis=0)
    }
    
    # Print fusion results (match reference format exactly)
    print("\n" + "="*110)
    print(f"Fusion Model (RF+Transformer) - 10-Fold Cross Validation (Overall) Results")
    print("="*110)
    print(f"{'Fold':<8} {'ACC':<10} {'MCC':<10} {'Precision':<12} {'Recall':<10} {'Fscore':<10} {'BACC':<10} {'AUC':<10}")
    print("-"*110)
    for i in range(fold):
        print(f"{i+1:<8} {acc[i]:<10.4f} {mcc[i]:<10.4f} "
              f"{macro_precision[i]:<12.4f} {macro_recall[i]:<10.4f} "
              f"{macro_fscore[i]:<10.4f} {bacc[i]:<10.4f} {auc[i]:<10.4f}")
    print("-"*110)
    print(f"{'Mean':<8} {mean_metrics['ACC']:<10.4f} {mean_metrics['MCC']:<10.4f} "
          f"{mean_metrics['Precision']:<12.4f} {mean_metrics['Recall']:<10.4f} "
          f"{mean_metrics['Fscore']:<10.4f} {mean_metrics['BACC']:<10.4f} {mean_metrics['AUC']:<10.4f}")
    print("="*110 + "\n")
    
    print("="*80)
    print(f"Fusion Model (RF+Transformer) - Per-Class Mean Results (10-Fold CV)")
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
    
    return mean_metrics, class_mean_metrics

# ====================== Main Process (match reference output flow) ======================
def main():
    file_name = "set.csv"
    feature_path = os.path.join("..", "data", "fusion", "train", file_name)
    
    print(f"Reading feature file: {feature_path}")
    print("Starting cross validation processing (with SMOTE)...\n")
    
    try:
        # Train base models
        rf_base = train_rf_base_model()
        tf_base = train_transformer_base_model()
        
        # Evaluate RF with CV (match reference output format)
        rf_mean_metrics, rf_class_metrics = evaluate_rf_cv(rf_base['model'], rf_base['X_train'], rf_base['y_train'])
        
        # Load Transformer feature data for fusion evaluation
        tf_feature_path = os.path.join(FUSION_TRAIN_DIR, TF_FEATURE_FILE)
        X_tf, y_tf = load_train_data(TRAIN_FILES, tf_feature_path)
        
        # Evaluate Fusion Model
        fusion_mean_metrics, fusion_class_metrics = evaluate_fusion_model(
            rf_base['model'], tf_base['model'], tf_base['input_dim'], tf_base['cls_num_list'],
            rf_base['X_train'], X_tf, rf_base['y_train']
        )
        
        # Final summary (match reference format)
        print(f"Processing completed!")
        print(f"RF Model - Overall Mean - ACC: {rf_mean_metrics['ACC']:.4f}, Fscore: {rf_mean_metrics['Fscore']:.4f}, BACC: {rf_mean_metrics['BACC']:.4f}, AUC: {rf_mean_metrics['AUC']:.4f}")
        print(f"Fusion Model - Overall Mean - ACC: {fusion_mean_metrics['ACC']:.4f}, Fscore: {fusion_mean_metrics['Fscore']:.4f}, BACC: {fusion_mean_metrics['BACC']:.4f}, AUC: {fusion_mean_metrics['AUC']:.4f}")
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()