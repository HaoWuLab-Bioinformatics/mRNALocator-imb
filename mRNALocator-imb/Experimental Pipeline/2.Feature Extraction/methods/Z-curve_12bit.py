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
def numba_calc_zcurve_12bit(seq: str) -> np.ndarray:
    seq_len = len(seq)
    if seq_len < 2:
        return np.zeros(12, dtype=np.float64)
    total_pairs = seq_len - 1
    
    AA = 0
    AC = 0
    AG = 0
    AT = 0
    CA = 0
    CC = 0
    CG = 0
    CT = 0
    GA = 0
    GC = 0
    GG = 0
    GT = 0
    TA = 0
    TC = 0
    TG = 0
    TT = 0
    
    for i in prange(seq_len - 1):
        c1 = seq[i]
        c2 = seq[i+1]
        if c1 == 'A':
            if c2 == 'A':
                AA +=1
            elif c2 == 'C':
                AC +=1
            elif c2 == 'G':
                AG +=1
            elif c2 == 'T':
                AT +=1
        elif c1 == 'C':
            if c2 == 'A':
                CA +=1
            elif c2 == 'C':
                CC +=1
            elif c2 == 'G':
                CG +=1
            elif c2 == 'T':
                CT +=1
        elif c1 == 'G':
            if c2 == 'A':
                GA +=1
            elif c2 == 'C':
                GC +=1
            elif c2 == 'G':
                GG +=1
            elif c2 == 'T':
                GT +=1
        elif c1 == 'T':
            if c2 == 'A':
                TA +=1
            elif c2 == 'C':
                TC +=1
            elif c2 == 'G':
                TG +=1
            elif c2 == 'T':
                TT +=1
    
    z_vals = np.zeros(12, dtype=np.float64)
    z_vals[0] = (AA + AG - AC - AT) / total_pairs
    z_vals[1] = (AA + AC - AG - AT) / total_pairs
    z_vals[2] = (AA + AT - AG - AC) / total_pairs
    z_vals[3] = (CA + CG - CC - CT) / total_pairs
    z_vals[4] = (CA + CC - CG - CT) / total_pairs
    z_vals[5] = (CA + CT - CG - CC) / total_pairs
    z_vals[6] = (GA + GG - GC - GT) / total_pairs
    z_vals[7] = (GA + GC - GG - GT) / total_pairs
    z_vals[8] = (GA + GT - GG - GC) / total_pairs
    z_vals[9] = (TA + TG - TC - TT) / total_pairs
    z_vals[10] = (TA + TC - TG - TT) / total_pairs
    z_vals[11] = (TA + TT - TG - TC) / total_pairs
    
    return z_vals

def Z_curve_12bit(sequences, path, dataset_type: str = 'train'):
    NN = 'ACGT'
    encodings = []
    header = ['SampleName']
    for base in NN:
        for elem in ['x', 'y', 'z']:
            header.append('%s.%s' % (base, elem))
    encodings.append(header)

    i = -1
    print(f"making {dataset_type.upper()} Z-curve_12bit feature...")
    for sequence in tqdm(sequences):
        i += 1
        code = [f"seq{i}"]
        z_vals = numba_calc_zcurve_12bit(sequence)
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

method = 'Z-curve_12bit'
feature_method = Z_curve_12bit

train_feature = process_feature(train_mRna_seq, 'train', method, feature_method)
test_feature = process_feature(test_mRna_seq, 'test', method, feature_method)
weight_feature = process_feature(weight_mRna_seq, 'weight', method, feature_method)

print("\nZ-curve_12bit feature extraction done.")
print(f"Train feature shape: {train_feature.shape}")
print(f"Test feature shape: {test_feature.shape}")
print(f"Weight feature shape: {weight_feature.shape}")