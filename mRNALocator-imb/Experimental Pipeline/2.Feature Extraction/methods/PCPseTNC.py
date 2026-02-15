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

myTriIndex = {
    'AAA': 0, 'AAC': 1, 'AAG': 2, 'AAT': 3,
    'ACA': 4, 'ACC': 5, 'ACG': 6, 'ACT': 7,
    'AGA': 8, 'AGC': 9, 'AGG': 10, 'AGT': 11,
    'ATA': 12, 'ATC': 13, 'ATG': 14, 'ATT': 15,
    'CAA': 16, 'CAC': 17, 'CAG': 18, 'CAT': 19,
    'CCA': 20, 'CCC': 21, 'CCG': 22, 'CCT': 23,
    'CGA': 24, 'CGC': 25, 'CGG': 26, 'CGT': 27,
    'CTA': 28, 'CTC': 29, 'CTG': 30, 'CTT': 31,
    'GAA': 32, 'GAC': 33, 'GAG': 34, 'GAT': 35,
    'GCA': 36, 'GCC': 37, 'GCG': 38, 'GCT': 39,
    'GGA': 40, 'GGC': 41, 'GGG': 42, 'GGT': 43,
    'GTA': 44, 'GTC': 45, 'GTG': 46, 'GTT': 47,
    'TAA': 48, 'TAC': 49, 'TAG': 50, 'TAT': 51,
    'TCA': 52, 'TCC': 53, 'TCG': 54, 'TCT': 55,
    'TGA': 56, 'TGC': 57, 'TGG': 58, 'TGT': 59,
    'TTA': 60, 'TTC': 61, 'TTG': 62, 'TTT': 63
}

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
def numba_get_kmer_frequency(seq_num_arr: np.ndarray) -> np.ndarray:
    seq_len = seq_num_arr.shape[0]
    kmer = 3
    freq = np.zeros(64, dtype=np.float64)
    total = seq_len - kmer + 1
    if total <= 0:
        return freq
    for i in range(total):
        idx = 16 * seq_num_arr[i] + 4 * seq_num_arr[i+1] + seq_num_arr[i+2]
        freq[idx] += 1
    freq /= total
    return freq

@njit(cache=True, nogil=True, fastmath=True, parallel=True)
def numba_get_theta_array(seq_num_arr: np.ndarray, prop_arr: np.ndarray, lamadaValue: int) -> np.ndarray:
    seq_len = seq_num_arr.shape[0]
    kmer = 3
    theta_array = np.zeros(lamadaValue, dtype=np.float64)
    prop_len = prop_arr.shape[0]
    for tmpLamada in prange(lamadaValue):
        theta = 0.0
        valid_i = seq_len - tmpLamada - kmer
        if valid_i <= 0:
            theta_array[tmpLamada] = 0.0
            continue
        for i in range(valid_i):
            idx1 = 16 * seq_num_arr[i] + 4 * seq_num_arr[i+1] + seq_num_arr[i+2]
            idx2 = 16 * seq_num_arr[i + tmpLamada + 1] + 4 * seq_num_arr[i + tmpLamada + 2] + seq_num_arr[i + tmpLamada + 3]
            cc = 0.0
            for p in range(prop_len):
                cc += (prop_arr[p][idx1] - prop_arr[p][idx2]) ** 2
            cc /= prop_len
            theta += cc
        theta /= valid_i
        theta_array[tmpLamada] = theta
    return theta_array

def make_PCPseTNC_vector(sequences, path, dataset_type: str = 'train', lamadaValue: int = 10, weight: float = 0.1):
    file_name = './data/tridnaPhyche.data'
    with open(file_name, 'rb') as f:
        myPropertyValue = pickle.load(f)
    
    myPropertyName = ['Dnase I', 'Bendability (DNAse)']
    
    prop_arr = np.array([myPropertyValue[p] for p in myPropertyName], dtype=np.float64)
    
    encodings = []
    header = ['SampleName']
    for tripeptide in sorted(myTriIndex.keys()):
        header.append(tripeptide)
    for k in range(1, lamadaValue + 1):
        header.append('lamada_' + str(k))
    encodings.append(header)
    
    print(f"making {dataset_type.upper()} PCPseTNC feature...")
    i = -1
    for sequence in tqdm(sequences):
        i += 1
        code = [f"seq{i}"]
        seq_num_arr = dna_str2num_arr(sequence)
        tripeptideFrequency = numba_get_kmer_frequency(seq_num_arr)
        thetaArray = numba_get_theta_array(seq_num_arr, prop_arr, lamadaValue)
        
        sum_theta = np.sum(thetaArray)
        denom = 1 + weight * sum_theta
        
        freq_part = tripeptideFrequency / denom
        code.extend(freq_part.tolist())
        
        theta_part = (weight * thetaArray) / denom
        code.extend(theta_part.tolist())
        
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

method = 'PCPseTNC'
feature_method = make_PCPseTNC_vector

train_feature = process_feature(train_mRna_seq, 'train', method, feature_method)
test_feature = process_feature(test_mRna_seq, 'test', method, feature_method)
weight_feature = process_feature(weight_mRna_seq, 'weight', method, feature_method)

print("\nPCPseTNC feature extraction done.")
print(f"Train feature shape: {train_feature.shape}")
print(f"Test feature shape: {test_feature.shape}")
print(f"Weight feature shape: {weight_feature.shape}")