import warnings
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

def read_fasta(file_path):
    sequences = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            seq = ''
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if seq:
                        sequences.append(seq)
                        seq = ''
                else:
                    seq += line
            if seq:
                sequences.append(seq)
    except Exception as e:
        print(f"Error reading FASTA file ({file_path}): {e}")
    return sequences

train_file_name = ['../data/train/Cytoplasm.fasta', '../data/train/Endoplasmic_reticulum.fasta',
                   '../data/train/Extracellular_region.fasta', '../data/train/Mitochondria.fasta',
                   '../data/train/Nucleus.fasta']

test_file_name = ['../data/test/Cytoplasm.fasta',
                  '../data/test/Endoplasmic_reticulum.fasta',
                  '../data/test/Extracellular_region.fasta',
                  '../data/test/Mitochondria.fasta',
                  '../data/test/Nucleus.fasta']

train_mRna_label = []
for file in train_file_name:
    seq = read_fasta(file)
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
    train_mRna_label += label

test_mRna_label = []
for file in test_file_name:
    seq = read_fasta(file)
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
    test_mRna_label += label

train_feature_path = '../data/fusion/train/set.csv'
test_feature_path = '../data/fusion/test/set.csv'

with open(train_feature_path, 'r', encoding='utf-8') as f:
    header_line = f.readline().strip()
    columns = [col.strip() for col in header_line.split(',') if col.strip()]
print(f"Number of feature names extracted from header: {len(columns)}")

train_df = pd.read_csv(
    train_feature_path, sep=',', low_memory=False,
    header=0, index_col=None
)
test_df = pd.read_csv(
    test_feature_path, sep=',', low_memory=False,
    header=0, index_col=None
)

X = train_df.iloc[:, 1:].astype(np.float64).values
y = np.array(train_mRna_label, dtype=int)
x_test = test_df.iloc[:, 1:].astype(np.float64).values
y_test = np.array(test_mRna_label, dtype=int)

print(f"Training feature dimension (samples, features): {X.shape}")
print(f"Test feature dimension (samples, features): {x_test.shape}")
print(f"Number of training labels: {len(y)}, Number of test labels: {len(y_test)}")

feature_columns = columns[1:]

if len(feature_columns) != X.shape[1]:
    raise ValueError(
        f"Feature names count ({len(feature_columns)}) does not match feature columns count ({X.shape[1]}). "
        f"Please check the CSV header and data columns."
    )

columns = feature_columns
print(f"Feature names count ({len(columns)}) matches feature columns count ({X.shape[1]}).")

model = RandomForestClassifier(
    n_estimators=250,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=False,
    random_state=10,
    n_jobs=-1
)
model.fit(X, y)
print(f"Model training completed, number of classes: {len(model.classes_)}")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_test)

print(f"\nSHAP values type: {type(shap_values)}")
if isinstance(shap_values, np.ndarray):
    print(f"SHAP values array dimension: {shap_values.shape}")
elif isinstance(shap_values, list):
    print(f"SHAP values list length (number of classes): {len(shap_values)}")
    if len(shap_values) > 0:
        print(f"Single class SHAP values dimension (samples, features): {shap_values[0].shape}")

class_names = ['Cyt', 'Endo', 'Extra', 'Mito', 'Nucl']
n_classes = len(class_names)
n_features = X.shape[1]

class_shap_mean = np.zeros((n_classes, n_features))
total_shap_importance = np.zeros(n_features)

if isinstance(shap_values, list) and len(shap_values) == n_classes:
    for cls_idx in range(n_classes):
        cls_shap = shap_values[cls_idx]
        cls_mean = np.mean(cls_shap, axis=0)
        class_shap_mean[cls_idx] = cls_mean
        total_shap_importance += np.abs(cls_mean)

elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    if shap_values.shape == (x_test.shape[0], n_features, n_classes):
        for cls_idx in range(n_classes):
            cls_shap = shap_values[:, :, cls_idx]
            cls_mean = np.mean(cls_shap, axis=0)
            class_shap_mean[cls_idx] = cls_mean
            total_shap_importance += np.abs(cls_mean)
    else:
        raise ValueError(
            f"SHAP values 3D array dimension mismatch, expected ({x_test.shape[0]}, {n_features}, {n_classes}), actual {shap_values.shape}")
else:
    raise ValueError(
        f"SHAP values structure abnormal, cannot process: type={type(shap_values)}, dimension={getattr(shap_values, 'shape', 'list length=' + str(len(shap_values)))}")

top_k = 30
sorted_indices = np.argsort(total_shap_importance)[::-1]
top_indices = sorted_indices[:min(top_k, len(sorted_indices))]

print("\n" + "=" * 130)
print(f"Top {top_k} Features (Total Impact Descending) - 6 decimal places:")
print("Ranking rule: Total Impact = |Cyt SHAP mean| + |Endo SHAP mean| + |Extra SHAP mean| + |Mito SHAP mean| + |Nucl SHAP mean|")
print("=" * 130)

header = (
    f"{'Rank':<6} {'Feature Name':<45} {'Total Impact':<12} "
    f"{'Cyt Mean':<12} {'Endo Mean':<12} {'Extra Mean':<12} {'Mito Mean':<12} {'Nucl Mean':<12}"
)
print(header)
print("-" * 130)

for rank, feat_idx in enumerate(top_indices, 1):
    feat_name = columns[feat_idx] if feat_idx < len(columns) else f"Feature_{feat_idx}"

    total_imp = round(float(total_shap_importance[feat_idx]), 6)

    cyt_mean = round(float(class_shap_mean[0, feat_idx]), 6)
    endo_mean = round(float(class_shap_mean[1, feat_idx]), 6)
    extra_mean = round(float(class_shap_mean[2, feat_idx]), 6)
    mito_mean = round(float(class_shap_mean[3, feat_idx]), 6)
    nucl_mean = round(float(class_shap_mean[4, feat_idx]), 6)

    row = (
        f"{rank:<6} {feat_name:<45} {total_imp:<12.6f} "
        f"{cyt_mean:<12.6f} {endo_mean:<12.6f} {extra_mean:<12.6f} {mito_mean:<12.6f} {nucl_mean:<12.6f}"
    )
    print(row)