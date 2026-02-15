import warnings
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

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
    train_feature = np.array(train_feature, dtype=np.float32)
    
    if len(train_labels) != train_feature.shape[0]:
        raise ValueError(f"Training label count ({len(train_labels)}) does not match feature count ({train_feature.shape[0]})!")
    
    return train_feature, train_labels

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=5, dropout=0.2):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers>1 else 0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, h_n = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.softmax(out)
        return out

class EarlyStopping:
    def __init__(self, patience=5, delta=1e-4, mode='max', verbose=False):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_params = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_params = model.state_dict()
        elif (self.mode == 'max' and score < self.best_score + self.delta) or \
             (self.mode == 'min' and score > self.best_score - self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_params = model.state_dict()
            self.counter = 0
        return self.early_stop

def train_gru_with_amp_earlystop(model, train_loader, val_loader, criterion, optimizer, device, epochs=50):
    model.train()
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=5, delta=1e-4, mode='max', verbose=False)

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch_y.numpy())
        
        val_mcc = matthews_corrcoef(val_labels, val_preds)
        if early_stopping(val_mcc, model):
            break

    model.load_state_dict(early_stopping.best_model_params)
    return model

def predict_gru(model, data_loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            with autocast():
                outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)

def train_cv_evaluate(X_train, y_train, fold=10, random_seed=10):
    acc = np.zeros(fold, dtype=float)
    mcc = np.zeros(fold, dtype=float)
    precision = np.zeros(fold, dtype=float)
    recall = np.zeros(fold, dtype=float)
    fscore = np.zeros(fold, dtype=float)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    hidden_dim = 64
    num_layers = 2
    num_classes = 5
    batch_size = 32
    max_epochs = 50
    lr = 0.001
    
    kf = KFold(n_splits=fold, shuffle=True, random_state=random_seed)
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    i = 0
    for train_index, val_index in tqdm(kf.split(X_train), desc=f"{fold}-Fold CV", total=fold, leave=True):
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]

        X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=random_seed, stratify=y_tr
        )

        X_train_sub_3d = X_train_sub.reshape(-1, 1, input_dim)
        X_val_sub_3d = X_val_sub.reshape(-1, 1, input_dim)
        X_val_3d = X_val.reshape(-1, 1, input_dim)

        train_dataset = TensorDataset(torch.from_numpy(X_train_sub_3d), torch.from_numpy(y_train_sub))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=4 if torch.cuda.is_available() else 0, 
                                  pin_memory=True if torch.cuda.is_available() else False)
        val_dataset = TensorDataset(torch.from_numpy(X_val_sub_3d), torch.from_numpy(y_val_sub))
        val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False,
                                num_workers=4 if torch.cuda.is_available() else 0, 
                                pin_memory=True if torch.cuda.is_available() else False)
        val_test_dataset = TensorDataset(torch.from_numpy(X_val_3d), torch.from_numpy(y_val))
        val_test_loader = DataLoader(val_test_dataset, batch_size=batch_size*2, shuffle=False,
                                     num_workers=4 if torch.cuda.is_available() else 0, 
                                     pin_memory=True if torch.cuda.is_available() else False)

        model = GRUClassifier(input_dim, hidden_dim, num_layers, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model = train_gru_with_amp_earlystop(model, train_loader, val_loader, criterion, optimizer, device, max_epochs)

        y_pred = predict_gru(model, val_test_loader, device)
        
        acc[i] = accuracy_score(y_val, y_pred)
        mcc[i] = matthews_corrcoef(y_val, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='macro')
        precision[i] = prec
        recall[i] = rec
        fscore[i] = f1
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