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

@njit(cache=True, nogil=True, fastmath=True)
def get_kmer_idx(seq_num_arr: np.ndarray, j: int, kmer: int) -> int:
    if kmer == 2:
        return 4 * seq_num_arr[j] + seq_num_arr[j+1]
    elif kmer == 3:
        return 16 * seq_num_arr[j] + 4 * seq_num_arr[j+1] + seq_num_arr[j+2]
    else:
        return 0

@njit(cache=True, nogil=True, fastmath=True, parallel=True)
def numba_calc_ac(seq_num_arr: np.ndarray, prop_arr: np.ndarray, kmer: int, lag_max: int) -> np.ndarray:
    seq_len = seq_num_arr.shape[0]
    kmer_num = seq_len - kmer + 1
    ac_vals = np.zeros(lag_max, dtype=np.float32)
    
    meanValue = 0.0
    for j in range(kmer_num):
        kmer_idx = get_kmer_idx(seq_num_arr, j, kmer)
        meanValue += prop_arr[kmer_idx]
    meanValue /= kmer_num
    
    for l in prange(1, lag_max + 1):
        acValue = 0.0
        valid_j = seq_len - kmer - l + 1
        if valid_j <= 0:
            ac_vals[l-1] = 0.0
            continue
        for j in range(valid_j):
            idx1 = get_kmer_idx(seq_num_arr, j, kmer)
            idx2 = get_kmer_idx(seq_num_arr, j+l, kmer)
            acValue += (prop_arr[idx1] - meanValue) * (prop_arr[idx2] - meanValue)
        acValue /= valid_j
        ac_vals[l-1] = acValue
    return ac_vals

@njit(cache=True, nogil=True, fastmath=True, parallel=True)
def numba_calc_cc(seq_num_arr: np.ndarray, prop1_arr: np.ndarray, prop2_arr: np.ndarray, kmer: int, lag_max: int) -> np.ndarray:
    seq_len = seq_num_arr.shape[0]
    kmer_num = seq_len - kmer + 1
    cc_vals = np.zeros(lag_max, dtype=np.float32)
    
    meanP1 = 0.0
    meanP2 = 0.0
    for j in range(kmer_num):
        kmer_idx = get_kmer_idx(seq_num_arr, j, kmer)
        meanP1 += prop1_arr[kmer_idx]
        meanP2 += prop2_arr[kmer_idx]
    meanP1 /= kmer_num
    meanP2 /= kmer_num
    
    for l in prange(1, lag_max + 1):
        ccValue = 0.0
        valid_j = seq_len - kmer - l + 1
        if valid_j <= 0:
            cc_vals[l-1] = 0.0
            continue
        for j in range(valid_j):
            idx1 = get_kmer_idx(seq_num_arr, j, kmer)
            idx2 = get_kmer_idx(seq_num_arr, j+l, kmer)
            ccValue += (prop1_arr[idx1] - meanP1) * (prop2_arr[idx2] - meanP2)
        ccValue /= valid_j
        cc_vals[l-1] = ccValue
    return cc_vals

def generatePropertyPairs(myPropertyName):
    pairs = []
    for i in range(len(myPropertyName)):
        for j in range(i + 1, len(myPropertyName)):
            pairs.append([myPropertyName[i], myPropertyName[j]])
            pairs.append([myPropertyName[j], myPropertyName[i]])
    return pairs

def make_ac_vector(sequences, myPropertyName, myPropertyValue, lag, kmer, path, dataset_type: str = 'train'):
    encodings = []
    header = ['SampleName']
    for p in myPropertyName:
        for l in range(1, lag + 1):
            header.append('%s.lag%d' % (p, l))
    encodings.append(header)
    
    prop_arr_dict = {p: np.array(myPropertyValue[p], dtype=np.float64) for p in myPropertyName}
    
    i = -1
    print(f"making {dataset_type.upper()} AC feature (lag={lag})...")
    for sequence in tqdm(sequences):
        i += 1
        code = [f"seq{i}"]
        seq_num_arr = dna_str2num_arr(sequence)
        
        for p in myPropertyName:
            ac_vals = numba_calc_ac(seq_num_arr, prop_arr_dict[p], kmer, lag)
            code.extend(ac_vals.tolist())
        
        encodings.append(code)
    
    save_to_csv(encodings, path)
    return encodings

def make_cc_vector(sequences, myPropertyName, myPropertyValue, lag, kmer, path, dataset_type: str = 'train'):
    encodings = []
    if len(myPropertyName) < 2:
        print('Error: two or more property are needed for cross covariance descriptors')
        exit(1)
    
    propertyPairs = generatePropertyPairs(myPropertyName)
    header = ['SampleName'] + [n[0] + '-' + n[1] + '-lag.' + str(l) for n in propertyPairs for l in range(1, lag + 1)]
    encodings.append(header)
    
    prop_arr_dict = {p: np.array(myPropertyValue[p], dtype=np.float64) for p in myPropertyName}
    
    i = -1
    print(f"making {dataset_type.upper()} CC feature (lag={lag})...")
    for sequence in tqdm(sequences):
        i += 1
        code = [f"seq{i}"]
        seq_num_arr = dna_str2num_arr(sequence)
        
        for pair in propertyPairs:
            p1, p2 = pair
            cc_vals = numba_calc_cc(seq_num_arr, prop_arr_dict[p1], prop_arr_dict[p2], kmer, lag)
            code.extend(cc_vals.tolist())
        
        encodings.append(code)
    
    save_to_csv(encodings, path)
    return encodings

def make_acc_vector(sequences, myPropertyName, myPropertyValue, lag, kmer, path, dataset_type: str = 'train'):
    encodings = []
    if len(myPropertyName) < 2:
        print('Error: two or more property are needed for cross covariance descriptors')
        exit(1)
    
    header = ['SampleName']
    for p in myPropertyName:
        for l in range(1, lag + 1):
            header.append('%s.lag%d' % (p, l))
    
    propertyPairs = generatePropertyPairs(myPropertyName)
    header += [n[0] + '-' + n[1] + '-lag.' + str(l) for n in propertyPairs for l in range(1, lag + 1)]
    encodings.append(header)
    
    prop_arr_dict = {p: np.array(myPropertyValue[p], dtype=np.float64) for p in myPropertyName}
    
    i = -1
    print(f"making {dataset_type.upper()} ACC feature (lag={lag})...")
    for sequence in tqdm(sequences):
        i += 1
        code = [f"seq{i}"]
        seq_num_arr = dna_str2num_arr(sequence)
        
        for p in myPropertyName:
            ac_vals = numba_calc_ac(seq_num_arr, prop_arr_dict[p], kmer, lag)
            code.extend(ac_vals.tolist())
        
        for pair in propertyPairs:
            p1, p2 = pair
            cc_vals = numba_calc_cc(seq_num_arr, prop_arr_dict[p1], prop_arr_dict[p2], kmer, lag)
            code.extend(cc_vals.tolist())
        
        encodings.append(code)
    
    save_to_csv(encodings, path)
    return encodings

myDictDefault = {
    'DAC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist']},
    'DCC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist']},
    'DACC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist']},
    'TAC': {'DNA': ['Dnase I', 'Bendability (DNAse)']},
    'TCC': {'DNA': ['Dnase I', 'Bendability (DNAse)']},
    'TACC': {'DNA': ['Dnase I', 'Bendability (DNAse)']}
}

myKmer = {
    'DAC': 2, 'DCC': 2, 'DACC': 2,
    'TAC': 3, 'TCC': 3, 'TACC': 3
}

myDataFile = {
    'DAC': {'DNA': 'didnaPhyche.data'},
    'DCC': {'DNA': 'didnaPhyche.data'},
    'DACC': {'DNA': 'didnaPhyche.data'},
    'TAC': {'DNA': 'tridnaPhyche.data'},
    'TCC': {'DNA': 'tridnaPhyche.data'},
    'TACC': {'DNA': 'tridnaPhyche.data'}
}

def check_acc_arguments(method, nctype):
    kmer = myKmer.get(method, 2)
    myIndex = myDictDefault[method][nctype]
    dataFile = myDataFile[method][nctype]

    if dataFile != '':
        data_path = './data'
        with open(os.path.join(data_path, dataFile), 'rb') as f:
            myProperty = pickle.load(f)

    if len(myIndex) == 0 or len(myProperty) == 0:
        print('Error: arguments is incorrect.')
        exit(1)
    return myIndex, myProperty, kmer

def get_seq_and_label(file_list):
    mRna_seq = []
    for file in file_list:
        seq = read_fasta(file)
        mRna_seq += seq
    return mRna_seq

def process_feature(seq_list, dataset_type, method, feature_method, **kwargs):
    cache_dir = f'../cache/{dataset_type}/'
    lag = kwargs.get('lag', 1)
    path = f'{cache_dir}{method}_lag={lag}.csv'
    
    if os.path.exists(path):
        print(f"Loading {dataset_type.upper()} {method.upper()} feature (lag={lag}) from cache...")
        feature = pd.read_csv(path, sep=',', low_memory=False, header=None, index_col=None).values.tolist()
    else:
        my_property_name, my_property_value, kmer = check_acc_arguments(method, 'DNA')
        feature = feature_method(seq_list, my_property_name, my_property_value, lag, kmer, path, dataset_type=dataset_type)
    
    return np.array(feature)

if __name__ == "__main__":
    train_file_name = ['../data/train/Cytoplasm.fasta', '../data/train/Endoplasmic_reticulum.fasta',
                       '../data/train/Extracellular_region.fasta', '../data/train/Mitochondria.fasta',
                       '../data/train/Nucleus.fasta']
    test_file_name = ['../data/test/Cytoplasm.fasta', '../data/test/Endoplasmic_reticulum.fasta',
                      '../data/test/Extracellular_region.fasta', '../data/test/Mitochondria.fasta',
                      '../data/test/Nucleus.fasta']
    weight_file_name = ['../data/weight/Cytoplasm.fasta', '../data/weight/Endoplasmic_reticulum.fasta',
                        '../data/weight/Extracellular_region.fasta', '../data/weight/Mitochondria.fasta',
                        '../data/weight/Nucleus.fasta']

    train_mRna_seq = get_seq_and_label(train_file_name)
    test_mRna_seq = get_seq_and_label(test_file_name)
    weight_mRna_seq = get_seq_and_label(weight_file_name)

    for dir_type in ['train', 'test', 'weight']:
        os.makedirs(f'../cache/{dir_type}/', exist_ok=True)

    method = 'DAC'
    if method[-3:] == 'ACC':
        feature_method = make_acc_vector
    elif method[-2:] == 'CC':
        feature_method = make_cc_vector
    elif method[-2:] == 'AC':
        feature_method = make_ac_vector
    else:
        print(f"Unsupported method: {method}")
        exit(1)

    for lag in range(1, 7):
        print(f"\nProcessing {method.upper()} with lag={lag}")
        train_feature = process_feature(train_mRna_seq, 'train', method, feature_method, lag=lag)
        test_feature = process_feature(test_mRna_seq, 'test', method, feature_method, lag=lag)
        weight_feature = process_feature(weight_mRna_seq, 'weight', method, feature_method, lag=lag)

        print(f"Train shape: {train_feature.shape}")
        print(f"Test shape: {test_feature.shape}")
        print(f"Weight shape: {weight_feature.shape}")

    print(f"\n{method.upper()} feature extraction (lag=1~6) done.")