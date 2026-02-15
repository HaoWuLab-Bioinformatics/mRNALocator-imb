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
def numba_calc_zcurve_36bit(seq: str) -> np.ndarray:
    seq_len = len(seq)
    if seq_len < 2:
        return np.zeros(36, dtype=np.float64)
    total_pairs = seq_len - 1
    
    pos1_AA = 0; pos1_AC = 0; pos1_AG = 0; pos1_AT = 0
    pos1_CA = 0; pos1_CC = 0; pos1_CG = 0; pos1_CT = 0
    pos1_GA = 0; pos1_GC = 0; pos1_GG = 0; pos1_GT = 0
    pos1_TA = 0; pos1_TC = 0; pos1_TG = 0; pos1_TT = 0
    
    pos2_AA = 0; pos2_AC = 0; pos2_AG = 0; pos2_AT = 0
    pos2_CA = 0; pos2_CC = 0; pos2_CG = 0; pos2_CT = 0
    pos2_GA = 0; pos2_GC = 0; pos2_GG = 0; pos2_GT = 0
    pos2_TA = 0; pos2_TC = 0; pos2_TG = 0; pos2_TT = 0
    
    pos3_AA = 0; pos3_AC = 0; pos3_AG = 0; pos3_AT = 0
    pos3_CA = 0; pos3_CC = 0; pos3_CG = 0; pos3_CT = 0
    pos3_GA = 0; pos3_GC = 0; pos3_GG = 0; pos3_GT = 0
    pos3_TA = 0; pos3_TC = 0; pos3_TG = 0; pos3_TT = 0
    
    for i in prange(seq_len - 1):
        c1 = seq[i]
        c2 = seq[i+1]
        pos_mod = (i + 1) % 3
        
        if pos_mod == 1:
            if c1 == 'A':
                if c2 == 'A': pos1_AA +=1
                elif c2 == 'C': pos1_AC +=1
                elif c2 == 'G': pos1_AG +=1
                elif c2 == 'T': pos1_AT +=1
            elif c1 == 'C':
                if c2 == 'A': pos1_CA +=1
                elif c2 == 'C': pos1_CC +=1
                elif c2 == 'G': pos1_CG +=1
                elif c2 == 'T': pos1_CT +=1
            elif c1 == 'G':
                if c2 == 'A': pos1_GA +=1
                elif c2 == 'C': pos1_GC +=1
                elif c2 == 'G': pos1_GG +=1
                elif c2 == 'T': pos1_GT +=1
            elif c1 == 'T':
                if c2 == 'A': pos1_TA +=1
                elif c2 == 'C': pos1_TC +=1
                elif c2 == 'G': pos1_TG +=1
                elif c2 == 'T': pos1_TT +=1
        elif pos_mod == 2:
            if c1 == 'A':
                if c2 == 'A': pos2_AA +=1
                elif c2 == 'C': pos2_AC +=1
                elif c2 == 'G': pos2_AG +=1
                elif c2 == 'T': pos2_AT +=1
            elif c1 == 'C':
                if c2 == 'A': pos2_CA +=1
                elif c2 == 'C': pos2_CC +=1
                elif c2 == 'G': pos2_CG +=1
                elif c2 == 'T': pos2_CT +=1
            elif c1 == 'G':
                if c2 == 'A': pos2_GA +=1
                elif c2 == 'C': pos2_GC +=1
                elif c2 == 'G': pos2_GG +=1
                elif c2 == 'T': pos2_GT +=1
            elif c1 == 'T':
                if c2 == 'A': pos2_TA +=1
                elif c2 == 'C': pos2_TC +=1
                elif c2 == 'G': pos2_TG +=1
                elif c2 == 'T': pos2_TT +=1
        elif pos_mod == 0:
            if c1 == 'A':
                if c2 == 'A': pos3_AA +=1
                elif c2 == 'C': pos3_AC +=1
                elif c2 == 'G': pos3_AG +=1
                elif c2 == 'T': pos3_AT +=1
            elif c1 == 'C':
                if c2 == 'A': pos3_CA +=1
                elif c2 == 'C': pos3_CC +=1
                elif c2 == 'G': pos3_CG +=1
                elif c2 == 'T': pos3_CT +=1
            elif c1 == 'G':
                if c2 == 'A': pos3_GA +=1
                elif c2 == 'C': pos3_GC +=1
                elif c2 == 'G': pos3_GG +=1
                elif c2 == 'T': pos3_GT +=1
            elif c1 == 'T':
                if c2 == 'A': pos3_TA +=1
                elif c2 == 'C': pos3_TC +=1
                elif c2 == 'G': pos3_TG +=1
                elif c2 == 'T': pos3_TT +=1
    
    z_vals = np.zeros(36, dtype=np.float64)
    idx = 0
    
    for base in ['A', 'C', 'G', 'T']:
        if base == 'A':
            p1_x = (pos1_AA + pos1_AG - pos1_AC - pos1_AT) / total_pairs
            p1_y = (pos1_AA + pos1_AC - pos1_AG - pos1_AT) / total_pairs
            p1_z = (pos1_AA + pos1_AT - pos1_AG - pos1_AC) / total_pairs
            p2_x = (pos2_AA + pos2_AG - pos2_AC - pos2_AT) / total_pairs
            p2_y = (pos2_AA + pos2_AC - pos2_AG - pos2_AT) / total_pairs
            p2_z = (pos2_AA + pos2_AT - pos2_AG - pos2_AC) / total_pairs
            p3_x = (pos3_AA + pos3_AG - pos3_AC - pos3_AT) / total_pairs
            p3_y = (pos3_AA + pos3_AC - pos3_AG - pos3_AT) / total_pairs
            p3_z = (pos3_AA + pos3_AT - pos3_AG - pos3_AC) / total_pairs
        elif base == 'C':
            p1_x = (pos1_CA + pos1_CG - pos1_CC - pos1_CT) / total_pairs
            p1_y = (pos1_CA + pos1_CC - pos1_CG - pos1_CT) / total_pairs
            p1_z = (pos1_CA + pos1_CT - pos1_CG - pos1_CC) / total_pairs
            p2_x = (pos2_CA + pos2_CG - pos2_CC - pos2_CT) / total_pairs
            p2_y = (pos2_CA + pos2_CC - pos2_CG - pos2_CT) / total_pairs
            p2_z = (pos2_CA + pos2_CT - pos2_CG - pos2_CC) / total_pairs
            p3_x = (pos3_CA + pos3_CG - pos3_CC - pos3_CT) / total_pairs
            p3_y = (pos3_CA + pos3_CC - pos3_CG - pos3_CT) / total_pairs
            p3_z = (pos3_CA + pos3_CT - pos3_CG - pos3_CC) / total_pairs
        elif base == 'G':
            p1_x = (pos1_GA + pos1_GG - pos1_GC - pos1_GT) / total_pairs
            p1_y = (pos1_GA + pos1_GC - pos1_GG - pos1_GT) / total_pairs
            p1_z = (pos1_GA + pos1_GT - pos1_GG - pos1_GC) / total_pairs
            p2_x = (pos2_GA + pos2_GG - pos2_GC - pos2_GT) / total_pairs
            p2_y = (pos2_GA + pos2_GC - pos2_GG - pos2_GT) / total_pairs
            p2_z = (pos2_GA + pos2_GT - pos2_GG - pos2_GC) / total_pairs
            p3_x = (pos3_GA + pos3_GG - pos3_GC - pos3_GT) / total_pairs
            p3_y = (pos3_GA + pos3_GC - pos3_GG - pos3_GT) / total_pairs
            p3_z = (pos3_GA + pos3_GT - pos3_GG - pos3_GC) / total_pairs
        elif base == 'T':
            p1_x = (pos1_TA + pos1_TG - pos1_TC - pos1_TT) / total_pairs
            p1_y = (pos1_TA + pos1_TC - pos1_TG - pos1_TT) / total_pairs
            p1_z = (pos1_TA + pos1_TT - pos1_TG - pos1_TC) / total_pairs
            p2_x = (pos2_TA + pos2_TG - pos2_TC - pos2_TT) / total_pairs
            p2_y = (pos2_TA + pos2_TC - pos2_TG - pos2_TT) / total_pairs
            p2_z = (pos2_TA + pos2_TT - pos2_TG - pos2_TC) / total_pairs
            p3_x = (pos3_TA + pos3_TG - pos3_TC - pos3_TT) / total_pairs
            p3_y = (pos3_TA + pos3_TC - pos3_TG - pos3_TT) / total_pairs
            p3_z = (pos3_TA + pos3_TT - pos3_TG - pos3_TC) / total_pairs
        
        z_vals[idx] = p1_x; idx +=1
        z_vals[idx] = p1_y; idx +=1
        z_vals[idx] = p1_z; idx +=1
        z_vals[idx] = p2_x; idx +=1
        z_vals[idx] = p2_y; idx +=1
        z_vals[idx] = p2_z; idx +=1
        z_vals[idx] = p3_x; idx +=1
        z_vals[idx] = p3_y; idx +=1
        z_vals[idx] = p3_z; idx +=1
    
    return z_vals

def Z_curve_36bit(sequences, path, dataset_type: str = 'train'):
    NN = 'ACGT'
    encodings = []
    header = ['SampleName']
    for base in NN:
        for pos in range(1, 4):
            for elem in ['x', 'y', 'z']:
                header.append('Pos_%s_%s.%s' % (pos, base, elem))
    encodings.append(header)

    i = -1
    print(f"making {dataset_type.upper()} Z-curve_36bit feature...")
    for sequence in tqdm(sequences):
        i += 1
        code = [f"seq{i}"]
        z_vals = numba_calc_zcurve_36bit(sequence)
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

method = 'Z-curve_36bit'
feature_method = Z_curve_36bit

train_feature = process_feature(train_mRna_seq, 'train', method, feature_method)
test_feature = process_feature(test_mRna_seq, 'test', method, feature_method)
weight_feature = process_feature(weight_mRna_seq, 'weight', method, feature_method)

print("\nZ-curve_36bit feature extraction done.")
print(f"Train feature shape: {train_feature.shape}")
print(f"Test feature shape: {test_feature.shape}")
print(f"Weight feature shape: {weight_feature.shape}")