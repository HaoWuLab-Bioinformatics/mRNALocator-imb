import os
import warnings
from numba import njit, prange
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

def read_fasta(file):
    seq_list = []
    current_seq = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n\r')
            if not line:
                continue
            if line.startswith('>'):
                if current_seq:
                    seq = ''.join(current_seq).replace(' ', '').upper()
                    seq_list.append(seq)
                    current_seq = []
            else:
                current_seq.append(line)
    if current_seq:
        seq = ''.join(current_seq).replace(' ', '').upper()
        seq_list.append(seq)
    return seq_list

def save_to_csv(encodings, file):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    csv_lines = (','.join(map(str, line)) for line in encodings)
    csv_content = '\n'.join(csv_lines)
    with open(file, 'w', encoding='utf-8') as f:
        f.write(csv_content)

@njit(cache=True, nogil=True, fastmath=True)
def numba_calc_zcurve_48bit(seq: str) -> np.ndarray:
    seq_len = len(seq)
    if seq_len < 3:
        return np.zeros(48, dtype=np.float64)
    total_tri = seq_len - 2
    
    counts = np.zeros(64, dtype=np.int64)
    for i in prange(seq_len - 2):
        c1 = seq[i]
        c2 = seq[i+1]
        c3 = seq[i+2]
        idx = 0
        if c1 == 'A': idx += 0
        elif c1 == 'C': idx += 16
        elif c1 == 'G': idx += 32
        elif c1 == 'T': idx += 48
        
        if c2 == 'A': idx += 0
        elif c2 == 'C': idx += 4
        elif c2 == 'G': idx += 8
        elif c2 == 'T': idx += 12
        
        if c3 == 'A': idx += 0
        elif c3 == 'C': idx += 1
        elif c3 == 'G': idx += 2
        elif c3 == 'T': idx += 3
        counts[idx] += 1
    
    z_vals = np.zeros(48, dtype=np.float64)
    idx = 0
    bases = ['A', 'C', 'G', 'T']
    for b1 in bases:
        for b2 in bases:
            b1_idx = 0 if b1 == 'A' else 16 if b1 == 'C' else 32 if b1 == 'G' else 48
            b2_idx = 0 if b2 == 'A' else 4 if b2 == 'C' else 8 if b2 == 'G' else 12
            
            a_count = counts[b1_idx + b2_idx + 0]
            c_count = counts[b1_idx + b2_idx + 1]
            g_count = counts[b1_idx + b2_idx + 2]
            t_count = counts[b1_idx + b2_idx + 3]
            
            x = (a_count + g_count - c_count - t_count) / total_tri
            y = (a_count + c_count - g_count - t_count) / total_tri
            z = (a_count + t_count - g_count - c_count) / total_tri
            
            z_vals[idx] = x
            z_vals[idx+1] = y
            z_vals[idx+2] = z
            idx += 3
    return z_vals

def Z_curve_48bit(sequences, path, dataset_type: str = 'train'):
    NN = 'ACGT'
    encodings = []
    header = ['SampleName']
    for base in NN:
        for base1 in NN:
            for elem in ['x', 'y', 'z']:
                header.append('%s%s.%s' % (base, base1, elem))
    encodings.append(header)

    i = -1
    print(f"making {dataset_type.upper()} Z-curve_48bit feature...")
    for sequence in tqdm(sequences):
        i += 1
        code = [f"seq{i}"]
        z_vals = numba_calc_zcurve_48bit(sequence)
        code.extend(z_vals.tolist())
        encodings.append(code)
    
    save_to_csv(encodings, path)
    return encodings

def get_seq_and_label(file_list):
    mRna_seq, mRna_label = [], []
    for file in file_list:
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
        mRna_seq += seq
        mRna_label += label
    return mRna_seq, mRna_label

def process_feature(seq_list, dataset_type, method, feature_method):
    cache_dir = f'../cache/{dataset_type}/'
    path = f'{cache_dir}{method}.csv'
    if os.path.exists(path):
        print(f"Loading {dataset_type.upper()} {method.upper()} feature from cache...")
        feature = pd.read_csv(path, sep=',', low_memory=False, header=None, index_col=None).values.tolist()
    else:
        feature = feature_method(seq_list, path, dataset_type=dataset_type)
    return np.array(feature)

train_file_name = ['../data/train/Cytoplasm.fasta', '../data/train/Endoplasmic_reticulum.fasta',
                   '../data/train/Extracellular_region.fasta', '../data/train/Mitochondria.fasta',
                   '../data/train/Nucleus.fasta']
test_file_name = ['../data/test/Cytoplasm.fasta', '../data/test/Endoplasmic_reticulum.fasta',
                  '../data/test/Extracellular_region.fasta', '../data/test/Mitochondria.fasta',
                  '../data/test/Nucleus.fasta']
weight_file_name = ['../data/weight/Cytoplasm.fasta', '../data/weight/Endoplasmic_reticulum.fasta',
                    '../data/weight/Extracellular_region.fasta', '../data/weight/Mitochondria.fasta',
                    '../data/weight/Nucleus.fasta']

train_mRna_seq, train_mRna_label = get_seq_and_label(train_file_name)
test_mRna_seq, test_mRna_label = get_seq_and_label(test_file_name)
weight_mRna_seq, weight_mRna_label = get_seq_and_label(weight_file_name)

for dir_type in ['train', 'test', 'weight']:
    os.makedirs(f'../cache/{dir_type}/', exist_ok=True)

method = 'Z-curve_48bit'
feature_method = Z_curve_48bit

train_feature = process_feature(train_mRna_seq, 'train', method, feature_method)
test_feature = process_feature(test_mRna_seq, 'test', method, feature_method)
weight_feature = process_feature(weight_mRna_seq, 'weight', method, feature_method)

print("\nZ-curve_48bit feature extraction done.")
print(f"Train feature shape: {train_feature.shape}")
print(f"Test feature shape: {test_feature.shape}")
print(f"Weight feature shape: {weight_feature.shape}")