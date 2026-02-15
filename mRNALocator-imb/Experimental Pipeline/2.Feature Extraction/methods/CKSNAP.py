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
def numba_calc_cksnap(seq_num_arr: np.ndarray, gap: int) -> np.ndarray:
    seq_len = seq_num_arr.shape[0]
    pair_count = np.zeros(16, dtype=np.int64)
    total = 0
    
    for index1 in prange(seq_len):
        index2 = index1 + gap + 1
        if index2 < seq_len:
            pair_idx = 4 * seq_num_arr[index1] + seq_num_arr[index2]
            pair_count[pair_idx] += 1
            total += 1
    
    if total == 0:
        return np.zeros(16, dtype=np.float64)
    return pair_count / total


def CKSNAP(sequences, gap, path, dataset_type: str = 'train'):
    AA = 'ACGT'
    encodings = []
    aaPairs = [aa1 + aa2 for aa1 in AA for aa2 in AA]

    header = ['SampleName']
    for g in range(gap + 1):
        for aa in aaPairs:
            header.append(aa + '.gap' + str(g))
    encodings.append(header)

    i = -1
    print(f"making {dataset_type.upper()} CKSNAP feature (gap={gap})...")
    for sequence in tqdm(sequences):
        i += 1
        code = [f"seq{i}"]
        seq_num_arr = dna_str2num_arr(sequence)
        for g in range(gap + 1):
            cksnap_vals = numba_calc_cksnap(seq_num_arr, g)
            code.extend(cksnap_vals.tolist())
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
    gap = kwargs.get('gap', 1)
    path = f'{cache_dir}{method}_gap={gap}.csv'
    if os.path.exists(path):
        print(f"Loading {dataset_type.upper()} {method.upper()} feature (gap={gap}) from cache...")
        feature = pd.read_csv(path, sep=',', low_memory=False, header=None, index_col=None).values.tolist()
    else:
        feature = feature_method(seq_list, gap, path, dataset_type=dataset_type)
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

method = 'CKSNAP'
feature_method = CKSNAP

for gap in range(1, 7):
    print(f"\nProcessing {method.upper()} with gap={gap}")
    train_feature = process_feature(train_mRna_seq, 'train', method, feature_method, gap=gap)
    test_feature = process_feature(test_mRna_seq, 'test', method, feature_method, gap=gap)
    weight_feature = process_feature(weight_mRna_seq, 'weight', method, feature_method, gap=gap)

    print(f"Train {method} feature (gap={gap}) shape: {train_feature.shape}")
    print(f"Test {method} feature (gap={gap}) shape: {test_feature.shape}")
    print(f"Weight {method} feature (gap={gap}) shape: {weight_feature.shape}")

print(f"\n{method.upper()} feature extraction (gap=1~6) done.")