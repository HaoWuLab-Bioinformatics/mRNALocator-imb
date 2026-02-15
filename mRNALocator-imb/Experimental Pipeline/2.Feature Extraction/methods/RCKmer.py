import itertools
import os
import warnings
from collections import Counter
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

@njit(cache=True, nogil=True)
def numba_rc(kmer):
    kmer_len = len(kmer)
    rc_chars = [''] * kmer_len
    for i in range(kmer_len):
        c = kmer[i]
        if c == 'A':
            rc_chars[kmer_len - 1 - i] = 'T'
        elif c == 'C':
            rc_chars[kmer_len - 1 - i] = 'G'
        elif c == 'G':
            rc_chars[kmer_len - 1 - i] = 'C'
        elif c == 'T':
            rc_chars[kmer_len - 1 - i] = 'A'
        else:
            rc_chars[kmer_len - 1 - i] = c
    return ''.join(rc_chars)

def generateRCKmer(kmerList):
    rckmerList = set()
    for kmer in kmerList:
        rckmer = numba_rc(kmer)
        rckmerList.add(sorted([kmer, rckmer])[0])
    return sorted(rckmerList)

def RCKmer(sequences, k, upto, normalize, path, dataset_type='train'):
    encoding = []
    header = ['SampleName']
    NA = 'ACGT'
    
    if upto == True:
        for tmpK in range(1, k + 1):
            tmpHeader = [''.join(kmer) for kmer in itertools.product(NA, repeat=tmpK)]
            header += generateRCKmer(tmpHeader)
        myDict = {}
        for kmer in header[1:]:
            rckmer = numba_rc(kmer)
            if kmer != rckmer:
                myDict[rckmer] = kmer
        encoding.append(header)
        i = -1
        print(f"making {dataset_type.upper()} RCKmer feature (k={k}, upto=True)...")
        for sequence in tqdm(sequences):
            count = Counter()
            for tmpK in range(1, k + 1):
                kmers = [sequence[i:i+tmpK] for i in range(len(sequence)-tmpK+1)]
                for j in range(len(kmers)):
                    if kmers[j] in myDict:
                        kmers[j] = myDict[kmers[j]]
                count.update(kmers)
                if normalize == True:
                    kmer_len = len(sequence) - tmpK + 1
                    if kmer_len > 0:
                        for key in count:
                            if len(key) == tmpK:
                                count[key] = count[key] / kmer_len
            i += 1
            code = [f"seq{i}"]
            for j in range(1, len(header)):
                code.append(count.get(header[j], 0))
            encoding.append(code)
    else:
        tmpHeader = [''.join(kmer) for kmer in itertools.product(NA, repeat=k)]
        header += generateRCKmer(tmpHeader)
        myDict = {}
        for kmer in header[1:]:
            rckmer = numba_rc(kmer)
            if kmer != rckmer:
                myDict[rckmer] = kmer

        encoding.append(header)
        i = -1
        print(f"making {dataset_type.upper()} RCKmer feature (k={k})...")
        for sequence in tqdm(sequences):
            seq_len = len(sequence)
            kmer_num = seq_len - k + 1
            if kmer_num <= 0:
                kmers = []
            else:
                kmers = [sequence[i:i+k] for i in range(kmer_num)]
            for j in range(len(kmers)):
                if kmers[j] in myDict:
                    kmers[j] = myDict[kmers[j]]
            count = Counter(kmers)
            if normalize == True and kmer_num > 0:
                for key in count:
                    count[key] = count[key] / kmer_num
            i += 1
            code = [f"seq{i}"]
            for j in range(1, len(header)):
                code.append(count.get(header[j], 0))
            encoding.append(code)
    save_to_csv(encoding, path)
    return encoding

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

def process_feature(seq_list, dataset_type, method, feature_method, k):
    cache_dir = f'../cache/{dataset_type}/'
    path = f'{cache_dir}{method}_k={k}.csv'
    if os.path.exists(path):
        print(f"Loading {dataset_type.upper()} {method.upper()} feature (k={k}) from cache...")
        feature = pd.read_csv(path, sep=',', low_memory=False, header=None, index_col=None).values.tolist()
    else:
        feature = feature_method(seq_list, k, False, True, path, dataset_type=dataset_type)
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

method = 'RCKmer'
feature_method = RCKmer

for k in range(2, 7):
    print(f"\nProcessing {method.upper()} with k={k}")
    train_feature = process_feature(train_mRna_seq, 'train', method, feature_method, k)
    test_feature = process_feature(test_mRna_seq, 'test', method, feature_method, k)
    weight_feature = process_feature(weight_mRna_seq, 'weight', method, feature_method, k)

    print(f"Train {method} feature (k={k}) shape: {train_feature.shape}")
    print(f"Test {method} feature (k={k}) shape: {test_feature.shape}")
    print(f"Weight {method} feature (k={k}) shape: {weight_feature.shape}")

print(f"\n{method.upper()} feature extraction (k=2~6) done.")