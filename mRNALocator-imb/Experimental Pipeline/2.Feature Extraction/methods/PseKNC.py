import os
import warnings
import itertools
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

myDiIndex = {
    'AA': 0, 'AC': 1, 'AG': 2, 'AT': 3,
    'CA': 4, 'CC': 5, 'CG': 6, 'CT': 7,
    'GA': 8, 'GC': 9, 'GG': 10, 'GT': 11,
    'TA': 12, 'TC': 13, 'TG': 14, 'TT': 15
}

baseSymbol = 'ACGT'

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
def numba_kmer_frequency(seq_num_arr: np.ndarray, kmer: int) -> np.ndarray:
    seq_len = seq_num_arr.shape[0]
    kmer_size = 4 ** kmer
    freq = np.zeros(kmer_size, dtype=np.float64)
    total = seq_len - kmer + 1
    if total <= 0:
        return freq
    for i in prange(seq_len - kmer + 1):
        idx = 0
        for j in range(kmer):
            idx = idx * 4 + seq_num_arr[i + j]
        freq[idx] += 1
    freq /= total
    return freq

@njit(cache=True, nogil=True, fastmath=True)
def numba_correlation_function(prop1: np.ndarray, prop2: np.ndarray, num_props: int) -> float:
    cc = 0.0
    for p in range(num_props):
        cc += (prop1[p] - prop2[p]) ** 2
    return cc / num_props

@njit(cache=True, nogil=True, fastmath=True, parallel=True)
def numba_get_theta_array(seq_num_arr: np.ndarray, prop_arr: np.ndarray, lamada_value: int, kmer: int) -> np.ndarray:
    seq_len = seq_num_arr.shape[0]
    num_props = prop_arr.shape[1]
    theta_array = np.zeros(lamada_value, dtype=np.float64)
    for tmp_lamada in prange(lamada_value):
        theta = 0.0
        valid_i = seq_len - tmp_lamada - kmer
        if valid_i <= 0:
            theta_array[tmp_lamada] = 0.0
            continue
        for i in range(valid_i):
            idx1 = 0
            for j in range(kmer):
                idx1 = idx1 * 4 + seq_num_arr[i + j]
            idx2 = 0
            for j in range(kmer):
                idx2 = idx2 * 4 + seq_num_arr[i + tmp_lamada + 1 + j]
            cc = numba_correlation_function(prop_arr[idx1], prop_arr[idx2], num_props)
            theta += cc
        theta_array[tmp_lamada] = theta / valid_i
    return theta_array

def check_pse_arguments(method, nctype, weight, kmer, lamadaValue):
    if not 0 < weight < 1:
        print('Error: the weight factor ranged from 0 ~ 1.')
        exit(1)
    if not 0 < kmer < 10:
        print('Error: the kmer value ranged from 1 - 10')
        exit(1)
    
    myDictDefault = {
        'PseKNC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
                   'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']}
    }
    
    myDataFile = {
        'PseKNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'}
    }
    
    myIndex = myDictDefault[method][nctype]
    dataFile = myDataFile[method][nctype]
    
    myProperty = {}
    if dataFile != '':
        with open('./data/' + dataFile, 'rb') as f:
            myProperty = pickle.load(f)
    
    if len(myIndex) == 0 or len(myProperty) == 0:
        print('Error: arguments is incorrect.')
        exit(1)
    
    return myIndex, myProperty, lamadaValue, weight, kmer

def make_PseKNC_vector(sequences, myPropertyName, myPropertyValue, lamadaValue, weight, kmer, path, dataset_type: str = 'train'):
    encodings = []
    kmer_list = [''.join(i) for i in itertools.product(baseSymbol, repeat=kmer)]
    kmer_list_sorted = sorted(kmer_list)
    
    header = ['SampleName'] + kmer_list_sorted
    for l in range(1, lamadaValue + 1):
        header.append(f'lamada_{l}')
    encodings.append(header)
    
    di_kmer_list = sorted(myDiIndex.keys())
    prop_arr = np.array([[myPropertyValue[p][myDiIndex[di_kmer]] for p in myPropertyName] for di_kmer in di_kmer_list], dtype=np.float64)
    
    i = -1
    print(f"making {dataset_type.upper()} PseKNC feature (k={kmer}, lamada={lamadaValue})...")
    for sequence in tqdm(sequences):
        i += 1
        code = [f"seq{i}"]
        seq_num_arr = dna_str2num_arr(sequence)
        
        kmer_freq = numba_kmer_frequency(seq_num_arr, kmer)
        
        theta_array = numba_get_theta_array(seq_num_arr, prop_arr, lamadaValue, 2)
        
        denominator = 1 + weight * np.sum(theta_array)
        freq_vals = kmer_freq / denominator
        code.extend(freq_vals.tolist())
        
        theta_vals = (weight * theta_array) / denominator
        code.extend(theta_vals.tolist())
        
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

def process_feature(seq_list, dataset_type, method, feature_method, k, lamadaValue, weight):
    cache_dir = f'../cache/{dataset_type}/'
    path = f'{cache_dir}{method}_k={k}.csv'
    if os.path.exists(path):
        print(f"Loading {dataset_type.upper()} {method.upper()} feature (k={k}) from cache...")
        feature = pd.read_csv(path, sep=',', low_memory=False, header=None, index_col=None).values.tolist()
    else:
        my_property_name, my_property_value, lamadaValue, weight, kmer = check_pse_arguments(method, 'DNA', weight, k, lamadaValue)
        feature = feature_method(seq_list, my_property_name, my_property_value, lamadaValue, weight, kmer, path, dataset_type=dataset_type)
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

method = 'PseKNC'
feature_method = make_PseKNC_vector
weight = 0.1
lamadaValue = 10

for k in range(2, 7):
    print(f"\nProcessing {method.upper()} with k={k}")
    train_feature = process_feature(train_mRna_seq, 'train', method, feature_method, k, lamadaValue, weight)
    test_feature = process_feature(test_mRna_seq, 'test', method, feature_method, k, lamadaValue, weight)
    weight_feature = process_feature(weight_mRna_seq, 'weight', method, feature_method, k, lamadaValue, weight)

    print(f"Train {method} feature (k={k}) shape: {train_feature.shape}")
    print(f"Test {method} feature (k={k}) shape: {test_feature.shape}")
    print(f"Weight {method} feature (k={k}) shape: {weight_feature.shape}")

print(f"\n{method.upper()} feature extraction (k=2~6) done.")