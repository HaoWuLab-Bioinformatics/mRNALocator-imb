import warnings
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, precision_recall_fscore_support,
    balanced_accuracy_score, roc_auc_score
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler

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
        raise ValueError("Training label count does not match feature count!")
    
    return train_feature, train_labels

def get_resampling_methods():
    resampling_methods = {
        "No Resampling": None,
        "SMOTE": SMOTE(random_state=10, k_neighbors=3, sampling_strategy="not minority"),
        "ADASYN": ADASYN(random_state=10, n_neighbors=3, sampling_strategy="minority"),
        "Random Oversampling": RandomOverSampler(random_state=10, sampling_strategy="not minority"),
        "NearMiss": NearMiss(version=1, sampling_strategy="not majority"),
        "Random Downsampling": RandomUnderSampler(random_state=10, sampling_strategy="not majority")
    }
    return resampling_methods

def train_cv_evaluate(X_train, y_train, resampler_name="No Resampling", resampler=None, fold=10, random_seed=10):
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
    
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        random_state=random_seed, 
        n_jobs=-1
    )
    kf = KFold(n_splits=fold, shuffle=True, random_state=random_seed)
    
    i = 0
    for train_index, val_index in tqdm(kf.split(X_train), desc=f"{resampler_name} - {fold}-Fold CV", total=fold, leave=True):
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]
        
        if resampler is not None:
            try:
                X_tr, y_tr = resampler.fit_resample(X_tr, y_tr)
            except ValueError as e:
                print(f"\nWarning: Fold {i+1} - {resampler_name} resampling failed. Use original data.")
                pass
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)
        
        acc[i] = accuracy_score(y_val, y_pred)
        mcc[i] = matthews_corrcoef(y_val, y_pred)
        bacc[i] = balanced_accuracy_score(y_val, y_pred)
        
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_val, y_pred, average='macro', zero_division=0)
        macro_precision[i] = prec_macro
        macro_recall[i] = rec_macro
        macro_fscore[i] = f1_macro
        
        prec_cls, rec_cls, f1_cls, _ = precision_recall_fscore_support(y_val, y_pred, average=None, zero_division=0)
        class_precision[i] = prec_cls
        class_recall[i] = rec_cls
        class_fscore[i] = f1_cls
        
        y_val_binarized = label_binarize(y_val, classes=range(NUM_CLASSES))
        if NUM_CLASSES == 2:
            auc[i] = roc_auc_score(y_val, y_pred_proba[:, 1])
        else:
            auc[i] = roc_auc_score(y_val_binarized, y_pred_proba, average='macro', multi_class='ovr')
        
        i += 1
    
    fold_metrics = {
        'ACC': acc, 'MCC': mcc, 'Precision': macro_precision, 
        'Recall': macro_recall, 'Fscore': macro_fscore, 'BACC': bacc, 'AUC': auc
    }
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
    file_name = "set.csv"
    feature_path = os.path.join("..", "data", "fusion", "train", file_name)
    fold = 10
    random_seed = 10
    
    print(f"Reading feature file: {feature_path}")
    print("Starting cross validation...\n")
    
    try:
        base_method, full_method = parse_feature_filename(file_name)
        X_train, y_train = load_train_data(TRAIN_FASTA_PATHS, feature_path)
        resampling_methods = get_resampling_methods()
        all_results = {}
        
        print("\n" + "="*120)
        print("Testing resampling methods (10-fold CV on training set)")
        print("="*120)
        
        for method_name, resampler in resampling_methods.items():
            print(f"\n{'='*80}")
            print(f"Processing {method_name}...")
            print(f"{'='*80}")
            
            fold_metrics, mean_metrics, class_mean_metrics = train_cv_evaluate(
                X_train, y_train, 
                resampler_name=method_name,
                resampler=resampler,
                fold=fold, 
                random_seed=random_seed
            )
            
            all_results[method_name] = {
                'fold_metrics': fold_metrics,
                'mean_metrics': mean_metrics,
                'class_mean_metrics': class_mean_metrics
            }
            
            print(f"\n{method_name} - {full_method} - 10-Fold Cross Validation (Overall) Results")
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
            
            print(f"{method_name} - {full_method} - Per-Class Mean Results (10-Fold CV)")
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
        print("Summary of all resampling methods (10-fold CV mean)")
        print("="*120)
        print(f"{'Resampling Method':<20} {'ACC':<10} {'MCC':<10} {'Fscore':<10} {'BACC':<10} {'AUC':<10}")
        print("-"*70)
        for method_name, results in all_results.items():
            mean_metrics = results['mean_metrics']
            print(f"{method_name:<20} {mean_metrics['ACC']:<10.4f} {mean_metrics['MCC']:<10.4f} "
                  f"{mean_metrics['Fscore']:<10.4f} {mean_metrics['BACC']:<10.4f} {mean_metrics['AUC']:<10.4f}")
        print("="*120 + "\n")
        
        print("Processing completed!")
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()