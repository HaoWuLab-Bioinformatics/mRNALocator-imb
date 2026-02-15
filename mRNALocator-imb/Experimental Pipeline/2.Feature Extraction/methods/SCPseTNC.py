import os
import warnings
import itertools
from numba import njit, prange
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

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
baseSymbol = 'ACGT'

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

def check_Pse_arguments(method, nctype, weight, kmer, lamadaValue):
    if not 0 < weight < 1:
        print('Error: the weight factor ranged from 0 ~ 1.')
        import sys
        sys.exit(1)
    if not 0 < kmer < 10:
        print('Error: the kmer value ranged from 1 - 10')
        import sys
        sys.exit(1)
    myDictDefault = {
        'SCPseTNC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []}
    }
    myDataFile = {
        'SCPseTNC': {'DNA': 'tridnaPhyche.data', 'RNA': ''}
    }
    myIndex = myDictDefault[method][nctype]
    dataFile = myDataFile[method][nctype]
    myProperty = {}
    if dataFile != '':
        import pickle
        with open('./data/' + dataFile, 'rb') as f:
            myProperty = pickle.load(f)
    if len(myIndex) == 0 or len(myProperty) == 0:
        print('Error: arguments is incorrect.')
        import sys
        sys.exit(1)
    return myIndex, myProperty, lamadaValue, weight, kmer

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
def numba_get_kmer_frequency(seq_arr, kmer, tri_index_len):
    seq_len = seq_arr.shape[0]
    kmer_num = seq_len - kmer + 1
    frequency = np.zeros(tri_index_len, dtype=np.float64)
    for i in range(kmer_num):
        if kmer == 3:
            idx = 16 * seq_arr[i] + 4 * seq_arr[i+1] + seq_arr[i+2]
            frequency[idx] += 1
    frequency /= kmer_num
    return frequency

@njit(cache=True, nogil=True, fastmath=True)
def numba_correlationFunction_type2(idx1, idx2, prop_arr):
    CC = 0.0
    prop_len = prop_arr.shape[0]
    for p in range(prop_len):
        CC += prop_arr[p][idx1] * prop_arr[p][idx2]
    return CC

@njit(cache=True, nogil=True, fastmath=True, parallel=True)
def numba_get_theta_array_type2(seq_num_arr, prop_arr, lamadaValue, kmer):
    seq_len = seq_num_arr.shape[0]
    prop_len = prop_arr.shape[0]
    thetaArray = np.zeros(lamadaValue * prop_len, dtype=np.float64)
    for tmpLamada in prange(lamadaValue):
        for p in range(prop_len):
            theta = 0.0
            valid_i = seq_len - tmpLamada - kmer
            for i in range(valid_i):
                idx1 = 16 * seq_num_arr[i] + 4 * seq_num_arr[i+1] + seq_num_arr[i+2]
                idx2 = 16 * seq_num_arr[i + tmpLamada + 1] + 4 * seq_num_arr[i + tmpLamada + 2] + seq_num_arr[i + tmpLamada + 3]
                theta += prop_arr[p][idx1] * prop_arr[p][idx2]
            if valid_i > 0:
                theta /= valid_i
            thetaArray[tmpLamada * prop_len + p] = theta
    return thetaArray

def make_SCPseTNC_vector(sequences, myPropertyName, myPropertyValue, lamadaValue, weight, path, dataset_type: str = 'train'):
    encodings = []
    header = ['SampleName']
    tri_index_sorted = sorted(myTriIndex)
    for pep in tri_index_sorted:
        header.append(pep)
    for k in range(1, lamadaValue * len(myPropertyName) + 1):
        header.append('lamada_' + str(k))
    encodings.append(header)
    prop_arr = np.array([myPropertyValue[p] for p in myPropertyName], dtype=np.float64)
    tri_index_len = len(myTriIndex)
    i = -1
    print(f"making {dataset_type.upper()} SCPseTNC feature (lamada={lamadaValue}, weight={weight})...")
    for sequence in tqdm(sequences):
        i += 1
        code = [f"seq{i}"]
        seq_num_arr = dna_str2num_arr(sequence)
        tripeptideFrequency = numba_get_kmer_frequency(seq_num_arr, 3, tri_index_len)
        thetaArray = numba_get_theta_array_type2(seq_num_arr, prop_arr, lamadaValue, 3)
        sum_theta = np.sum(thetaArray)
        denom = 1 + weight * sum_theta
        for freq in tripeptideFrequency:
            code.append(freq / denom)
        for theta in thetaArray:
            code.append((weight * theta) / denom)
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
    weight = kwargs.get('weight', 0.1)
    lamadaValue = kwargs.get('lamadaValue', 10)
    path = f'{cache_dir}{method}_lamada={lamadaValue}_weight={weight}.csv'
    if os.path.exists(path):
        print(f"Loading {dataset_type.upper()} {method.upper()} feature (lamada={lamadaValue}, weight={weight}) from cache...")
        feature = pd.read_csv(path, sep=',', low_memory=False, header=None, index_col=None).values.tolist()
    else:
        nctype = kwargs.get('nctype', 'DNA')
        kmer = kwargs.get('kmer', 3)
        my_property_name, my_property_value, lamadaValue, weight, kmer = check_Pse_arguments(method, nctype, weight, kmer, lamadaValue)
        feature = feature_method(seq_list, my_property_name, my_property_value, lamadaValue, weight, path, dataset_type=dataset_type)
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

method = 'SCPseTNC'
feature_method = make_SCPseTNC_vector
nctype = 'DNA'
weight = 0.1
kmer = 3
lamadaValue = 10

train_feature = process_feature(train_mRna_seq, 'train', method, feature_method, nctype=nctype, weight=weight, kmer=kmer, lamadaValue=lamadaValue)
test_feature = process_feature(test_mRna_seq, 'test', method, feature_method, nctype=nctype, weight=weight, kmer=kmer, lamadaValue=lamadaValue)
weight_feature = process_feature(weight_mRna_seq, 'weight', method, feature_method, nctype=nctype, weight=weight, kmer=kmer, lamadaValue=lamadaValue)

print("\nSCPseTNC feature extraction done.")
print(f"Train feature shape: {train_feature.shape}")
print(f"Test feature shape: {test_feature.shape}")
print(f"Weight feature shape: {weight_feature.shape}")