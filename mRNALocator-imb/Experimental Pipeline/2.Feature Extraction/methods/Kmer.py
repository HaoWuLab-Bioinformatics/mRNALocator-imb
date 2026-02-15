import os
import warnings
from numba import njit, prange
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
from collections import Counter

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
def dna_str2num_arr(seq: str) -> np.ndarray:
    seq_len = len(seq)
    num_arr = np.zeros(seq_len, dtype=np.int8)
    for i in range(seq_len):
        c = seq[i]
        if c == 'A':
            num_arr[i] = 0
        elif c == 'C':
            num_arr[i] = 1
        elif c == 'G':
            num_arr[i] = 2
        elif c == 'T':
            num_arr[i] = 3
    return num_arr

@njit(cache=True, nogil=True, fastmath=True)
def numba_calc_asdc(seq_num_arr: np.ndarray) -> np.ndarray:
    seq_len = seq_num_arr.shape[0]
    pair_count = np.zeros(16, dtype=np.int64)
    total = 0
    for j in prange(seq_len):
        for k in range(j + 1, seq_len):
            pair_idx = 4 * seq_num_arr[j] + seq_num_arr[k]
            pair_count[pair_idx] += 1
            total += 1
    asdc_vals = np.zeros(16, dtype=np.float64)
    if total != 0:
        asdc_vals = pair_count / total
    return asdc_vals

def ASDC(sequences, path, dataset_type: str = 'train'):
    AA = 'ACGT'
    encodings = []
    aaPairs = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['SampleName']
    header += aaPairs
    encodings.append(header)
    i = -1
    print(f"making {dataset_type.upper()} ASDC feature...")
    for sequence in tqdm(sequences):
        i += 1
        code = [f"seq{i}"]
        seq_num_arr = dna_str2num_arr(sequence)
        asdc_vals = numba_calc_asdc(seq_num_arr)
        code.extend(asdc_vals.tolist())
        encodings.append(code)
    save_to_csv(encodings, path)
    return encodings

@njit(cache=True, nogil=True, fastmath=True)
def numba_kmer_count(seq_num_arr: np.ndarray, k: int) -> np.ndarray:
    seq_len = len(seq_num_arr)
    kmer_num = 4 ** k
    count = np.zeros(kmer_num, dtype=np.float64)
    valid_len = seq_len - k + 1
    if valid_len > 0:
        for i in range(valid_len):
            idx = 0
            for j in range(k):
                idx = idx * 4 + seq_num_arr[i + j]
            count[idx] += 1 
        count = count / valid_len
    return count

def generate_kmer_header(k):
    NA = 'ACGT'
    header = ['SampleName']
    for kmer in itertools.product(NA, repeat=k):
        header.append(''.join(kmer))
    return header

def Kmer(sequences, k, path, dataset_type: str = 'train', normalize=True):
    header = generate_kmer_header(k)
    encodings = [header]
    kmer_num = 4 ** k
    i = -1
    print(f"making {dataset_type.upper()} Kmer feature (k={k})...")
    for sequence in tqdm(sequences):
        i += 1
        code = [f"seq{i}"]
        seq_num_arr = dna_str2num_arr(sequence)
        count = numba_kmer_count(seq_num_arr, k)
        code.extend(count.tolist())
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

def process_feature(seq_list, dataset_type, method, feature_method, **kwargs):
    cache_dir = f'../cache/{dataset_type}/'
    if method == 'Kmer':
        k = kwargs.get('k', 2)
        path = f'{cache_dir}{method}_k={k}.csv'
    else:
        path = f'{cache_dir}{method}.csv'
    if os.path.exists(path):
        if method == 'Kmer':
            print(f"Loading {dataset_type.upper()} {method.upper()} feature (k={k}) from cache...")
        else:
            print(f"Loading {dataset_type.upper()} {method.upper()} feature from cache...")
        feature = pd.read_csv(path, sep=',', low_memory=False, header=None, index_col=None).values.tolist()
    else:
        if method == 'Kmer':
            feature = feature_method(seq_list, k, path, dataset_type=dataset_type)
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

method = 'ASDC'
feature_method = ASDC

train_feature_asdc = process_feature(train_mRna_seq, 'train', method, feature_method)
test_feature_asdc = process_feature(test_mRna_seq, 'test', method, feature_method)
weight_feature_asdc = process_feature(weight_mRna_seq, 'weight', method, feature_method)

print("\nASDC feature extraction done.")
print(f"Train ASDC feature shape: {train_feature_asdc.shape}")
print(f"Test ASDC feature shape: {test_feature_asdc.shape}")
print(f"Weight ASDC feature shape: {weight_feature_asdc.shape}")

method = 'Kmer'
feature_method = Kmer

for k in range(2, 9):
    print(f"\nProcessing {method.upper()} with k={k}")
    train_feature_kmer = process_feature(train_mRna_seq, 'train', method, feature_method, k=k)
    test_feature_kmer = process_feature(test_mRna_seq, 'test', method, feature_method, k=k)
    weight_feature_kmer = process_feature(weight_mRna_seq, 'weight', method, feature_method, k=k)

    print(f"Train {method} feature (k={k}) shape: {train_feature_kmer.shape}")
    print(f"Test {method} feature (k={k}) shape: {test_feature_kmer.shape}")
    print(f"Weight {method} feature (k={k}) shape: {weight_feature_kmer.shape}")

print(f"\n{method.upper()} feature extraction (k=2~10) done.")