import os
import warnings
from numba import njit, prange
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

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

def generatePropertyPairs(myPropertyName):
    pairs = []
    for i in range(len(myPropertyName)):
        for j in range(i + 1, len(myPropertyName)):
            pairs.append([myPropertyName[i], myPropertyName[j]])
            pairs.append([myPropertyName[j], myPropertyName[i]])
    return pairs

@njit(cache=True, nogil=True, fastmath=True, parallel=True)
def numba_calc_ac(seq, prop_arr, my_index_vals, kmer, lag_max, seq_len):
    ac_vals = np.zeros(lag_max, dtype=np.float32)
    kmer_num = seq_len - kmer + 1
    meanValue = 0.0
    for j in range(kmer_num):
        kmer_idx = my_index_vals[j]
        meanValue += prop_arr[kmer_idx]
    meanValue /= kmer_num
    for l in prange(1, lag_max + 1):
        acValue = 0.0
        valid_j = seq_len - kmer - l + 1
        for j in range(valid_j):
            idx1 = my_index_vals[j]
            idx2 = my_index_vals[j + l]
            acValue += (prop_arr[idx1] - meanValue) * (prop_arr[idx2] - meanValue)
        if valid_j > 0:
            acValue /= valid_j
        ac_vals[l-1] = acValue
    return ac_vals

@njit(cache=True, nogil=True, fastmath=True, parallel=True)
def numba_calc_cc(seq, prop1_arr, prop2_arr, my_index_vals, kmer, lag_max, seq_len):
    cc_vals = np.zeros(lag_max, dtype=np.float32)
    kmer_num = seq_len - kmer + 1
    meanP1 = 0.0
    meanP2 = 0.0
    for j in range(kmer_num):
        kmer_idx = my_index_vals[j]
        meanP1 += prop1_arr[kmer_idx]
        meanP2 += prop2_arr[kmer_idx]
    meanP1 /= kmer_num
    meanP2 /= kmer_num
    for l in prange(1, lag_max + 1):
        ccValue = 0.0
        valid_j = seq_len - kmer - l + 1
        for j in range(valid_j):
            idx1 = my_index_vals[j]
            idx2 = my_index_vals[j + l]
            ccValue += (prop1_arr[idx1] - meanP1) * (prop2_arr[idx2] - meanP2)
        if valid_j > 0:
            ccValue /= valid_j
        cc_vals[l-1] = ccValue
    return cc_vals

def make_acc_vector(sequences, myPropertyName, myPropertyValue, lag, kmer, path, dataset_type='train'):
    encodings = []
    myIndex = myDiIndex if kmer == 2 else myTriIndex
    if len(myPropertyName) < 2:
        print('Error: two or more property are needed for cross covariance (i.e. DCC and TCC) descriptors')
        exit(1)
    header = ['SampleName']
    for p in myPropertyName:
        for l in range(1, lag + 1):
            header.append('%s.lag%d' % (p, l))
    propertyPairs = generatePropertyPairs(myPropertyName)
    header = header + [n[0] + '-' + n[1] + '-lag.' + str(l) for n in propertyPairs for l in range(1, lag + 1)]
    encodings.append(header)
    prop_arr_dict = {p: np.array(myPropertyValue[p], dtype=np.float64) for p in myPropertyName}
    i = -1
    print(f"making {dataset_type.upper()} ACC feature (lag={lag})...")
    for sequence in tqdm(sequences):
        i += 1
        code = [f"seq{i}"]
        seq_len = len(sequence)
        kmer_num = seq_len - kmer + 1
        my_index_vals = np.zeros(kmer_num, dtype=np.int32)
        for j in range(kmer_num):
            kmer_seq = sequence[j:j+kmer]
            my_index_vals[j] = myIndex[kmer_seq]
        for p in myPropertyName:
            ac_vals = numba_calc_ac(sequence, prop_arr_dict[p], my_index_vals, kmer, lag, seq_len)
            code.extend(ac_vals.tolist())
        for pair in propertyPairs:
            p1, p2 = pair
            cc_vals = numba_calc_cc(sequence, prop_arr_dict[p1], prop_arr_dict[p2], my_index_vals, kmer, lag, seq_len)
            code.extend(cc_vals.tolist())
        encodings.append(code)
    save_to_csv(encodings, path)
    return encodings

def make_cc_vector(sequences, myPropertyName, myPropertyValue, lag, kmer, path, dataset_type='train'):
    encodings = []
    myIndex = myDiIndex if kmer == 2 else myTriIndex
    if len(myPropertyName) < 2:
        print('Error: two or more property are needed for cross covariance (i.e. DCC and TCC) descriptors')
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
        seq_len = len(sequence)
        kmer_num = seq_len - kmer + 1
        my_index_vals = np.zeros(kmer_num, dtype=np.int32)
        for j in range(kmer_num):
            kmer_seq = sequence[j:j+kmer]
            my_index_vals[j] = myIndex[kmer_seq]
        for pair in propertyPairs:
            p1, p2 = pair
            cc_vals = numba_calc_cc(sequence, prop_arr_dict[p1], prop_arr_dict[p2], my_index_vals, kmer, lag, seq_len)
            code.extend(cc_vals.tolist())
        encodings.append(code)
    save_to_csv(encodings, path)
    return encodings

def make_ac_vector(sequences, myPropertyName, myPropertyValue, lag, kmer, path, dataset_type='train'):
    encodings = []
    myIndex = myDiIndex if kmer == 2 else myTriIndex
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
        seq_len = len(sequence)
        kmer_num = seq_len - kmer + 1
        my_index_vals = np.zeros(kmer_num, dtype=np.int32)
        for j in range(kmer_num):
            kmer_seq = sequence[j:j+kmer]
            my_index_vals[j] = myIndex[kmer_seq]
        for p in myPropertyName:
            ac_vals = numba_calc_ac(sequence, prop_arr_dict[p], my_index_vals, kmer, lag, seq_len)
            code.extend(ac_vals.tolist())
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
    kmer = myKmer[method]
    myIndex = []
    myProperty = {}
    dataFile = ''
    myIndex = myDictDefault[method][nctype]
    dataFile = myDataFile[method][nctype]
    if dataFile != '':
        with open('./data/' + dataFile, 'rb') as f:
            myProperty = pickle.load(f)
    if len(myIndex) == 0 or len(myProperty) == 0:
        print('Error: arguments is incorrect.')
        exit(1)
    return myIndex, myProperty, kmer

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
    lag = kwargs.get('lag', 1)
    path = f'{cache_dir}{method}_lag={lag}.csv'
    if os.path.exists(path):
        print(f"Loading {dataset_type.upper()} {method.upper()} feature (lag={lag}) from cache...")
        feature = pd.read_csv(path, sep=',', low_memory=False, header=None, index_col=None).values.tolist()
    else:
        my_property_name, my_property_value, kmer = check_acc_arguments(method, 'DNA')
        feature = feature_method(seq_list, my_property_name, my_property_value, lag, kmer, path, dataset_type=dataset_type)
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

method = 'DCC'
if method[-3:] == 'ACC':
    feature_method = make_acc_vector
elif method[-2:] == 'CC':
    feature_method = make_cc_vector
elif method[-2:] == 'AC':
    feature_method = make_ac_vector

for lag in range(1, 7):
    print(f"\nProcessing {method.upper()} with lag={lag}")
    train_feature = process_feature(train_mRna_seq, 'train', method, feature_method, lag=lag)
    test_feature = process_feature(test_mRna_seq, 'test', method, feature_method, lag=lag)
    weight_feature = process_feature(weight_mRna_seq, 'weight', method, feature_method, lag=lag)

    print(f"Train {method} feature (lag={lag}) shape: {train_feature.shape}")
    print(f"Test {method} feature (lag={lag}) shape: {test_feature.shape}")
    print(f"Weight {method} feature (lag={lag}) shape: {weight_feature.shape}")

print(f"\n{method.upper()} feature extraction (lag=1~6) done.")