import warnings
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import random

warnings.filterwarnings("ignore")

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

def calculate_and_print_label_distribution(fasta_paths):
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
    
    total_samples = len(train_labels)
    print("="*60)
    print("Dataset Class Distribution")
    print("="*60)
    for label in sorted(LABEL_MAP.keys()):
        count = np.sum(train_labels == label)
        ratio = count / total_samples * 100
        print(f"{LABEL_MAP[label]:<25} | Count: {count:<6} | Ratio: {ratio:.2f}%")
    print(f"{'Total':<25} | Count: {total_samples:<6} | Ratio: 100.00%")
    print("="*60 + "\n")

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
    
    return train_feature, train_labels

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
    precision, recall, fscore, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    
    if need_probs:
        all_probs = torch.cat(all_probs_gpu).cpu().numpy()
        return total_loss, acc, mcc, precision, recall, fscore, all_preds, all_labels, all_probs
    else:
        return total_loss, acc, mcc, precision, recall, fscore, all_preds, all_labels, None

def init_model(num_classes=5, input_dim=16):
    model = DNA_Transformer_Model(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    return model, criterion, optimizer, scheduler

def train_cv_evaluate(X_train, y_train, fold=10, random_seed=10):
    acc = np.zeros(fold, dtype=float)
    mcc = np.zeros(fold, dtype=float)
    precision = np.zeros(fold, dtype=float)
    recall = np.zeros(fold, dtype=float)
    fscore = np.zeros(fold, dtype=float)
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed) if torch.cuda.is_available() else None
    torch.cuda.manual_seed_all(random_seed) if torch.cuda.is_available() else None
    random.seed(random_seed)
    
    input_dim = X_train.shape[1]
    kf = KFold(n_splits=fold, shuffle=True, random_state=random_seed)
    
    i = 0
    for train_index, val_index in tqdm(kf.split(X_train), desc=f"{fold}-Fold CV", total=fold, leave=True):
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
        
        model, criterion, optimizer, scheduler = init_model(num_classes=5, input_dim=input_dim)
        best_val_acc = 0.0
        early_stop_count = 0
        best_model_state = None
        
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_mcc, val_p, val_r, val_f, _, _, _ = validate(
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
        _, test_acc, test_mcc, test_p, test_r, test_f, _, _, _ = validate(
            model, test_loader_fold, criterion, device, need_probs=False
        )
        
        acc[i] = test_acc
        mcc[i] = test_mcc
        precision[i] = test_p
        recall[i] = test_r
        fscore[i] = test_f
        i += 1
    
    fold_metrics = {
        'ACC': acc, 'MCC': mcc, 'Precision': precision, 'Recall': recall, 'Fscore': fscore
    }
    mean_metrics = {
        'ACC': np.mean(acc), 'MCC': np.mean(mcc),
        'Precision': np.mean(precision), 'Recall': np.mean(recall), 'Fscore': np.mean(fscore)
    }
    return fold_metrics, mean_metrics

def parse_feature_filename(file_name):
    name_without_ext = file_name[:-4]
    if '_' in name_without_ext:
        base_method = name_without_ext.split('_')[0]
        full_method = name_without_ext
    else:
        base_method = name_without_ext
        full_method = name_without_ext
    return base_method, full_method

def save_cv_result(fold_metrics, full_method, save_dir='../record/'):
    os.makedirs(os.path.join(save_dir, 'CV'), exist_ok=True)
    
    CV_Fold = [f'Fold={i+1}' for i in range(len(fold_metrics['ACC']))] + ['Mean']
    cv_record = pd.DataFrame({
        'CV_Fold': CV_Fold,
        'ACC': list(fold_metrics['ACC']) + [np.mean(fold_metrics['ACC'])],
        'MCC': list(fold_metrics['MCC']) + [np.mean(fold_metrics['MCC'])],
        'Precision': list(fold_metrics['Precision']) + [np.mean(fold_metrics['Precision'])],
        'Recall': list(fold_metrics['Recall']) + [np.mean(fold_metrics['Recall'])],
        'Fscore': list(fold_metrics['Fscore']) + [np.mean(fold_metrics['Fscore'])]
    })
    cv_record.to_csv(os.path.join(save_dir, f'CV/{full_method}_record.csv'), index=False)

def save_train_result(base_method_results, base_method, save_dir='../record/'):
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    
    train_record = pd.DataFrame({
        'Method': [item['full_method'] for item in base_method_results],
        'ACC': [item['mean_metrics']['ACC'] for item in base_method_results],
        'MCC': [item['mean_metrics']['MCC'] for item in base_method_results],
        'Precision': [item['mean_metrics']['Precision'] for item in base_method_results],
        'Recall': [item['mean_metrics']['Recall'] for item in base_method_results],
        'Fscore': [item['mean_metrics']['Fscore'] for item in base_method_results]
    })
    train_record.to_csv(os.path.join(save_dir, f'train/{base_method}_record.csv'), index=False)

def main():
    FEATURE_DIR = '../cache/train/'
    fold = 10
    random_seed = 10
    
    feature_files = [f for f in os.listdir(FEATURE_DIR) if f.endswith('.csv')]
    feature_files.sort()
    print(f"Found {len(feature_files)} feature files in {FEATURE_DIR} (sorted alphabetically)")
    
    calculate_and_print_label_distribution(TRAIN_FASTA_PATHS)
    
    print("Starting batch processing...\n")
    
    base_method_dict = {}
    
    for idx, file_name in enumerate(feature_files, 1):
        try:
            base_method, full_method = parse_feature_filename(file_name)
            feature_path = os.path.join(FEATURE_DIR, file_name)
            
            print(f"\nProcessing {idx}/{len(feature_files)}: {file_name} (Base method: {base_method})")
            
            X_train, y_train = load_train_data(TRAIN_FASTA_PATHS, feature_path)
            fold_metrics, mean_metrics = train_cv_evaluate(X_train, y_train, fold, random_seed)
            
            save_cv_result(fold_metrics, full_method)
            
            if base_method not in base_method_dict:
                base_method_dict[base_method] = []
            base_method_dict[base_method].append({
                'full_method': full_method,
                'mean_metrics': mean_metrics
            })
            
            print(f"Completed {file_name} - Mean ACC: {mean_metrics['ACC']:.4f}, Mean Fscore: {mean_metrics['Fscore']:.4f}")
        
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue
    
    print("\nSaving merged train results for each base method...")
    for base_method, results in base_method_dict.items():
        save_train_result(results, base_method)
        print(f"  Saved merged results for {base_method} → train/{base_method}_record.csv")
    
    print("\n" + "="*80)
    print("Batch processing finished. Summary:")
    print("="*80)
    print(f"- Processed {len(feature_files)} feature files")
    print(f"- Total base methods processed: {len(base_method_dict)}")
    print(f"  Base methods list: {', '.join(sorted(base_method_dict.keys()))}")

if __name__ == "__main__":
    main()