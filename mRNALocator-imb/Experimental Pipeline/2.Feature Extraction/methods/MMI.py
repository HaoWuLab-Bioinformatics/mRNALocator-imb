import os
import warnings
import math
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

NA = 'ACGT'
dinucleotide_list = [a1 + a2 for a1 in NA for a2 in NA]
trinucleotide_list = [a1 + a2 + a3 for a1 in NA for a2 in NA for a3 in NA]

sorted_dinuc_keys = sorted(list(set([''.join(sorted(elem)) for elem in dinucleotide_list])))
sorted_trinuc_keys = sorted(list(set([''.join(sorted(elem)) for elem in trinucleotide_list])))

def str2num_arr(seq: str) -> np.ndarray:
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

def calc_mmi(sequence: str) -> np.ndarray:
    seq_len = len(sequence)
    if seq_len < 3:
        return np.zeros(len(sorted_dinuc_keys) + len(sorted_trinuc_keys), dtype=np.float64)
    
    f1_dict = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    for elem in sequence:
        if elem in f1_dict:
            f1_dict[elem] += 1
    for key in f1_dict:
        f1_dict[key] /= seq_len
    
    f2_dict = {k: 0 for k in sorted_dinuc_keys}
    for i in range(seq_len - 1):
        dinuc = ''.join(sorted(sequence[i:i+2]))
        if dinuc in f2_dict:
            f2_dict[dinuc] += 1
    for key in f2_dict:
        if seq_len - 1 > 0:
            f2_dict[key] /= (seq_len - 1)
    
    f3_dict = {k: 0 for k in sorted_trinuc_keys}
    for i in range(seq_len - 2):
        trinuc = ''.join(sorted(sequence[i:i+3]))
        if trinuc in f3_dict:
            f3_dict[trinuc] += 1
    for key in f3_dict:
        if seq_len - 2 > 0:
            f3_dict[key] /= (seq_len - 2)
    
    mmi_dinuc = []
    for key in sorted_dinuc_keys:
        if f2_dict[key] > 1e-10 and f1_dict[key[0]] * f1_dict[key[1]] > 1e-10:
            val = f2_dict[key] * math.log(f2_dict[key] / (f1_dict[key[0]] * f1_dict[key[1]]))
        else:
            val = 0.0
        mmi_dinuc.append(val)
    
    mmi_trinuc = []
    for key in sorted_trinuc_keys:
        element1 = 0.0
        element2 = 0.0
        element3 = 0.0
        
        dinuc1 = ''.join(sorted(key[0:2]))
        if f2_dict[dinuc1] > 1e-10 and f1_dict[key[0]] * f1_dict[key[1]] > 1e-10:
            element1 = f2_dict[dinuc1] * math.log(f2_dict[dinuc1] / (f1_dict[key[0]] * f1_dict[key[1]]))
        
        dinuc2 = ''.join(sorted(key[0] + key[2]))
        if f2_dict[dinuc2] > 1e-10 and f1_dict[key[2]] > 1e-10:
            element2 = (f2_dict[dinuc2] / f1_dict[key[2]]) * math.log(f2_dict[dinuc2] / f1_dict[key[2]])
        
        dinuc3 = ''.join(sorted(key[1:3]))
        if f2_dict[dinuc3] > 1e-10 and f3_dict[key] / f2_dict[dinuc3] > 1e-10:
            element3 = (f3_dict[key] / f2_dict[dinuc3]) * math.log(f3_dict[key] / f2_dict[dinuc3])
        
        mmi_trinuc.append(element1 + element2 - element3)
    
    mmi_vals = np.array(mmi_dinuc + mmi_trinuc, dtype=np.float64)
    return mmi_vals

def MMI(sequences, path, dataset_type: str = 'train'):
    encodings = []
    header = ['SampleName']
    header += ['MMI_%s' % elem for elem in sorted_dinuc_keys]
    header += ['MMI_%s' % elem for elem in sorted_trinuc_keys]
    encodings.append(header)

    i = -1
    print(f"making {dataset_type.upper()} MMI feature...")
    for sequence in tqdm(sequences):
        i += 1
        code = [f"seq{i}"]
        
        mmi_vals = calc_mmi(sequence)
        code.extend(mmi_vals.tolist())
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

method = 'MMI'
feature_method = MMI

train_feature = process_feature(train_mRna_seq, 'train', method, feature_method)
test_feature = process_feature(test_mRna_seq, 'test', method, feature_method)
weight_feature = process_feature(weight_mRna_seq, 'weight', method, feature_method)

print("\nMMI feature extraction done.")
print(f"Train feature shape: {train_feature.shape}")
print(f"Test feature shape: {test_feature.shape}")
print(f"Weight feature shape: {weight_feature.shape}")