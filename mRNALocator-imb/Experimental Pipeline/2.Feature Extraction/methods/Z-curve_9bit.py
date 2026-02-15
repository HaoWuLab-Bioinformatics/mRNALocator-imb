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
def numba_calc_zcurve(seq: str) -> np.ndarray:
    seq_len = len(seq)
    pos1_A = 0
    pos1_C = 0
    pos1_G = 0
    pos1_T = 0
    pos2_A = 0
    pos2_C = 0
    pos2_G = 0
    pos2_T = 0
    pos3_A = 0
    pos3_C = 0
    pos3_G = 0
    pos3_T = 0
    
    for i in prange(seq_len):
        c = seq[i]
        mod = (i + 1) % 3
        if mod == 1:
            if c == 'A':
                pos1_A += 1
            elif c == 'C':
                pos1_C += 1
            elif c == 'G':
                pos1_G += 1
            elif c == 'T':
                pos1_T += 1
        elif mod == 2:
            if c == 'A':
                pos2_A += 1
            elif c == 'C':
                pos2_C += 1
            elif c == 'G':
                pos2_G += 1
            elif c == 'T':
                pos2_T += 1
        elif mod == 0:
            if c == 'A':
                pos3_A += 1
            elif c == 'C':
                pos3_C += 1
            elif c == 'G':
                pos3_G += 1
            elif c == 'T':
                pos3_T += 1
    
    z_vals = np.zeros(9, dtype=np.float64)
    if seq_len == 0:
        return z_vals
    
    z_vals[0] = (pos1_A + pos1_G - pos1_C - pos1_T) / seq_len
    z_vals[1] = (pos1_A + pos1_C - pos1_G - pos1_T) / seq_len
    z_vals[2] = (pos1_A + pos1_T - pos1_G - pos1_C) / seq_len
    z_vals[3] = (pos2_A + pos2_G - pos2_C - pos2_T) / seq_len
    z_vals[4] = (pos2_A + pos2_C - pos2_G - pos2_T) / seq_len
    z_vals[5] = (pos2_A + pos2_T - pos2_G - pos2_C) / seq_len
    z_vals[6] = (pos3_A + pos3_G - pos3_C - pos3_T) / seq_len
    z_vals[7] = (pos3_A + pos3_C - pos3_G - pos3_T) / seq_len
    z_vals[8] = (pos3_A + pos3_T - pos3_G - pos3_C) / seq_len
    
    return z_vals

def Z_curve_9bit(sequences, path, dataset_type: str = 'train'):
    encodings = []
    header = ['SampleName']
    for pos in range(1, 4):
        for elem in ['x', 'y', 'z']:
            header.append('Pos_%s.%s' % (pos, elem))
    encodings.append(header)

    i = -1
    print(f"making {dataset_type.upper()} Z-curve_9bit feature...")
    for sequence in tqdm(sequences):
        i += 1
        code = [f"seq{i}"]
        z_vals = numba_calc_zcurve(sequence)
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

method = 'Z-curve_9bit'
feature_method = Z_curve_9bit

train_feature = process_feature(train_mRna_seq, 'train', method, feature_method)
test_feature = process_feature(test_mRna_seq, 'test', method, feature_method)
weight_feature = process_feature(weight_mRna_seq, 'weight', method, feature_method)

print("\nZ-curve_9bit feature extraction done.")
print(f"Train feature shape: {train_feature.shape}")
print(f"Test feature shape: {test_feature.shape}")
print(f"Weight feature shape: {weight_feature.shape}")