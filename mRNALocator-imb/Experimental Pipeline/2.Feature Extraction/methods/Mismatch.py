import os
import warnings
from numba import njit, prange
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools

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
def numba_mismatch_count(seq1_num: np.ndarray, seq2_num: np.ndarray) -> int:
    mismatch = 0
    min_len = min(len(seq1_num), len(seq2_num))
    for i in range(min_len):
        if seq1_num[i] != seq2_num[i]:
            mismatch += 1
    return mismatch

@njit(cache=True, nogil=True, fastmath=True)
def get_kmer_num(idx: int, k: int) -> np.ndarray:
    kmer_num = np.zeros(k, dtype=np.int8)
    tmp = idx
    for i in range(k-1, -1, -1):
        kmer_num[i] = tmp % 4
        tmp = tmp // 4
    return kmer_num

@njit(cache=True, nogil=True, fastmath=True)
def numba_calc_mismatch(seq_num_arr: np.ndarray, k: int, m: int) -> np.ndarray:
    seq_len = len(seq_num_arr)
    kmer_total = 4 ** k
    count = np.zeros(kmer_total, dtype=np.float64)
    valid_kmer = seq_len - k + 1
    
    if valid_kmer > 0 and seq_len > 0:
        for idx in range(kmer_total):
            kmer_num = get_kmer_num(idx, k)
            for j in range(valid_kmer):
                current_kmer = seq_num_arr[j:j+k]
                if numba_mismatch_count(current_kmer, kmer_num) <= m:
                    count[idx] += 1
        
        count = count / seq_len
    
    return count

def generate_kmer_header(k):
    NN = 'ACGT'
    header = ['SampleName']
    for kmer in itertools.product(NN, repeat=k):
        header.append(''.join(kmer))
    return header

def Mismatch(sequences, k, m, path, dataset_type: str = 'train'):
    header = generate_kmer_header(k)
    encodings = [header]
    i = -1
    print(f"making {dataset_type.upper()} Mismatch feature (k={k}, m={m})...")
    for sequence in tqdm(sequences):
        i += 1
        code = [f"seq{i}"]
        seq_num_arr = dna_str2num_arr(sequence)
        count = numba_calc_mismatch(seq_num_arr, k, m)
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
    k = kwargs.get('k', 3)
    m = kwargs.get('m', 1)
    path = f'{cache_dir}{method}_k={k}_m={m}.csv'
    if os.path.exists(path):
        print(f"Loading {dataset_type.upper()} {method.upper()} feature (k={k}, m={m}) from cache...")
        feature = pd.read_csv(path, sep=',', low_memory=False, header=None, index_col=None).values.tolist()
    else:
        feature = feature_method(seq_list, k, m, path, dataset_type=dataset_type)
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

method = 'Mismatch'
feature_method = Mismatch
m_value = 1

for k in range(3, 7):
    print(f"\nProcessing {method.upper()} with k={k}, m={m_value}")
    train_feature = process_feature(train_mRna_seq, 'train', method, feature_method, k=k, m=m_value)
    test_feature = process_feature(test_mRna_seq, 'test', method, feature_method, k=k, m=m_value)
    weight_feature = process_feature(weight_mRna_seq, 'weight', method, feature_method, k=k, m=m_value)

    print(f"Train {method} feature (k={k}, m={m_value}) shape: {train_feature.shape}")
    print(f"Test {method} feature (k={k}, m={m_value}) shape: {test_feature.shape}")
    print(f"Weight {method} feature (k={k}, m={m_value}) shape: {weight_feature.shape}")

print(f"\n{method.upper()} feature extraction (k=3~6, m=1) done.")