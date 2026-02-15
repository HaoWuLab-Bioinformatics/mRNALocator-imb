import os
import warnings
import pickle
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

@njit(cache=True, nogil=True, fastmath=True, parallel=True)
def numba_calc_nmbroto(seq_num_arr: np.ndarray, prop_arr: np.ndarray, nlag: int) -> np.ndarray:
    seq_len = seq_num_arr.shape[0]
    N = seq_len - 1
    nmbroto_vals = np.zeros(nlag, dtype=np.float64)
    for d in prange(1, nlag + 1):
        atsd = 0.0
        count = 0
        if N > nlag:
            valid_j = N - d
            for j in range(valid_j):
                idx1 = 4 * seq_num_arr[j] + seq_num_arr[j+1]
                idx2 = 4 * seq_num_arr[j+d] + seq_num_arr[j+d+1]
                atsd += prop_arr[idx1] * prop_arr[idx2]
                count += 1
            if count > 0:
                atsd /= count
        nmbroto_vals[d-1] = atsd
    return nmbroto_vals

def NMBroto(sequences, path, dataset_type: str = 'train', nlag_: int = 2):
    sequence_type = 'DNA'
    if sequence_type == 'DNA':
        file_name = './data/didnaPhyche.data'
    else:
        file_name = './data/dirnaPhyche.data'

    with open(file_name, 'rb') as handle:
        property_dict = pickle.load(handle)
    property_name = list(property_dict.keys())

    for p_name in property_name:
        tmp = np.array(property_dict[p_name], dtype=float)
        pmean = np.average(tmp)
        pstd = np.std(tmp)
        property_dict[p_name] = (tmp - pmean) / pstd

    encodings = []
    header = ['SampleName']
    for p_name in property_name:
        for d in range(1, nlag_ + 1):
            header.append(p_name + '.lag' + str(d))
    encodings.append(header)

    i = -1
    print(f"making {dataset_type.upper()} NMBroto feature (nlag={nlag_})...")
    for sequence in tqdm(sequences):
        i += 1
        code = [f"seq{i}"]
        seq_num_arr = dna_str2num_arr(sequence)
        for p_name in property_name:
            prop_arr = np.array(property_dict[p_name], dtype=np.float64)
            nmbroto_vals = numba_calc_nmbroto(seq_num_arr, prop_arr, nlag_)
            code.extend(nmbroto_vals.tolist())
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
    nlag = kwargs.get('nlag', 2)
    path = f'{cache_dir}{method}_nlag={nlag}.csv'
    if os.path.exists(path):
        print(f"Loading {dataset_type.upper()} {method.upper()} feature (nlag={nlag}) from cache...")
        feature = pd.read_csv(path, sep=',', low_memory=False, header=None, index_col=None).values.tolist()
    else:
        feature = feature_method(seq_list, path, dataset_type=dataset_type, nlag_=nlag)
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

method = 'NMBroto'
feature_method = NMBroto

for nlag in range(2, 5):
    print(f"\nProcessing {method.upper()} with nlag={nlag}")
    train_feature = process_feature(train_mRna_seq, 'train', method, feature_method, nlag=nlag)
    test_feature = process_feature(test_mRna_seq, 'test', method, feature_method, nlag=nlag)
    weight_feature = process_feature(weight_mRna_seq, 'weight', method, feature_method, nlag=nlag)

    print(f"Train {method} feature (nlag={nlag}) shape: {train_feature.shape}")
    print(f"Test {method} feature (nlag={nlag}) shape: {test_feature.shape}")
    print(f"Weight {method} feature (nlag={nlag}) shape: {weight_feature.shape}")

print(f"\n{method.upper()} feature extraction (nlag=2~4) done.")