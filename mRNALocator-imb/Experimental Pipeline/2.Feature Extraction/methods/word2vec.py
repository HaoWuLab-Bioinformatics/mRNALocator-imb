import os
import warnings
from numba import njit, prange
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
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
def generate_kmer_indices(seq_len, k):
    indices = np.arange(seq_len - k + 1)
    return indices

def word_vector(sequences, kmers, vs, model, path, dataset_type: str = 'train'):
    encoding = []
    header = ['SampleName']
    for kmer in range(vs):
        header.append('W2V_Fea' + str(kmer))
    encoding.append(header)
    j = -1
    print(f"making {dataset_type.upper()} W2V feature (k={kmers})...")
    for sequence in tqdm(sequences):
        j += 1
        code = [f"seq{j}"]
        seq_len = len(sequence)
        if seq_len < kmers:
            vec = np.zeros(vs, dtype=np.float32)
            code.extend(vec.tolist())
            encoding.append(code)
            continue
        kmer_seq = [sequence[j:j + kmers] for j in range(seq_len - kmers + 1)]
        valid_kmers = [w for w in kmer_seq if w in model.wv]
        if not valid_kmers:
            vec = np.zeros(vs, dtype=np.float32)
        else:
            array = np.array([model.wv[w] for w in valid_kmers])
            idx = np.argwhere(np.all(array == 0, axis=1))
            array = np.delete(array, idx, axis=0) if idx.size > 0 else array
            vec = array.mean(axis=0) if array.size > 0 else np.zeros(vs, dtype=np.float32)
        code.extend(vec.tolist())
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

def process_feature(seq_list, dataset_type, k, ws, vs, model, feature_method):
    cache_dir = f'../cache/{dataset_type}/'
    method = f'W2V_k={k}_ws={ws}_vs={vs}'
    path = f'{cache_dir}{method}.csv'
    if os.path.exists(path):
        print(f"Loading {dataset_type.upper()} {method} feature from cache...")
        feature = pd.read_csv(path, sep=',', low_memory=False, header=None, index_col=None).values.tolist()
    else:
        feature = feature_method(seq_list, k, vs, model, path, dataset_type=dataset_type)
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
os.makedirs('./save/', exist_ok=True)

feature_method = word_vector
ws = 5
vs = 64

for k in range(3, 15):
    print(f"\nProcessing W2V feature with k={k}")
    RNADoc = [[i[j:j + k] for j in range(len(i) - k + 1)] for i in train_mRna_seq]
    doc = [i + ['<EOS>'] for i in RNADoc]
    wv_model = Word2Vec(
        doc,
        min_count=0,
        window=ws,
        vector_size=vs,
        workers=os.cpu_count(),
        sg=1,
        epochs=10)
    wv_model.save(f'./save/w2v_model_k={k}_ws={ws}_vs={vs}.w2v')
    
    train_feature = process_feature(train_mRna_seq, 'train', k, ws, vs, wv_model, feature_method)
    test_feature = process_feature(test_mRna_seq, 'test', k, ws, vs, wv_model, feature_method)
    weight_feature = process_feature(weight_mRna_seq, 'weight', k, ws, vs, wv_model, feature_method)

    print(f"Train W2V feature (k={k}) shape: {train_feature.shape}")
    print(f"Test W2V feature (k={k}) shape: {test_feature.shape}")
    print(f"Weight W2V feature (k={k}) shape: {weight_feature.shape}")

print("\nW2V feature extraction (k=3~14) done.")