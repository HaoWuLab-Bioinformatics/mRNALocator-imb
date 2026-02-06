import warnings

import numpy
import pandas
import shap
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from utils import read_fasta


def make_balanced_samples(features, labels, samples_per_class=100):
    unique_classes = np.unique(labels)
    balanced_features = []
    balanced_labels = []
    for cls in unique_classes:
        # 找到当前类别的所有样本
        indices = np.where(labels == cls)[0]
        # 如果当前类别样本数少于samples_per_class，则选择所有样本；否则随机选择samples_per_class个样本
        if len(indices) < samples_per_class:
            selected_indices = indices
        else:
            selected_indices = np.random.choice(indices, samples_per_class, replace=False)
        # 添加到平衡后的集合中
        balanced_features.extend(features[selected_indices])
        balanced_labels.extend(labels[selected_indices])
    return np.array(balanced_features), np.array(balanced_labels)


warnings.filterwarnings("ignore")

train_file_name = ['./data/train/Cytoplasm_train.fasta', './data/train/Endoplasmic_reticulum_train.fasta',
                   './data/train/Extracellular_region_train.fasta', './data/train/Mitochondria_train.fasta',
                   './data/train/Nucleus_train.fasta']

test_file_name = ['./data/indep_1/Cytoplasm_indep1.fasta',
                  './data/indep_1/Endoplasmic_reticulum_indep1.fasta',
                  './data/indep_1/Extracellular_region_indep1.fasta',
                  './data/indep_1/Mitochondria_indep1.fasta',
                  './data/indep_1/Nucleus_indep1.fasta']

train_mRna_seq = []
train_mRna_label = []
for file in train_file_name:
    seq = read_fasta(file)
    if file.find('Cytoplasm') > 0:
        label = [0] * len(seq)
    elif file.find('Endoplasmic') > 0:
        label = [1] * len(seq)
    elif file.find('Extracellula') > 0:
        label = [2] * len(seq)
    elif file.find('Mitochondria') > 0:
        label = [3] * len(seq)
    elif file.find('Nucleus') > 0:
        label = [4] * len(seq)
    train_mRna_seq = train_mRna_seq + seq
    train_mRna_label = train_mRna_label + label

test_mRna_seq = []
test_mRna_label = []
for file in test_file_name:
    seq = read_fasta(file)
    if file.find('Cytoplasm') > 0:
        label = [0] * len(seq)
    elif file.find('Endoplasmic') > 0:
        label = [1] * len(seq)
    elif file.find('Extracellula') > 0:
        label = [2] * len(seq)
    elif file.find('Mitochondria') > 0:
        label = [3] * len(seq)
    elif file.find('Nucleus') > 0:
        label = [4] * len(seq)
    test_mRna_seq = test_mRna_seq + seq
    test_mRna_label = test_mRna_label + label

train_feature = pandas.read_csv(
    './data/fusion/train/set.csv', sep=',', low_memory=False,
    header=None, index_col=None).values.tolist()

test_feature = pandas.read_csv(
    './data/fusion/test/set.csv', sep=',', low_memory=False,
    header=None, index_col=None).values.tolist()

train_feature = numpy.array(train_feature)
test_feature = numpy.array(test_feature)

X = train_feature[1:, 1:]
y = train_mRna_label
X = numpy.array(X, dtype=float)
y = numpy.array(y, dtype=int)
fold = 5
x_test = test_feature[1:, 1:]
y_test = numpy.array(test_mRna_label)
x_test = numpy.array(x_test, dtype=float)
y_test = numpy.array(y_test, dtype=int)

model = RandomForestClassifier(random_state=10)
model.fit(X, y)

columns = ['AAA_Bendability (DNAse) ', 'AAC_Bendability (DNAse) ', 'AAG_Bendability (DNAse) ',
           'AAT_Bendability (DNAse) ', 'ACA_Bendability (DNAse) ', 'ACC_Bendability (DNAse) ',
           'ACG_Bendability (DNAse) ', 'ACT_Bendability (DNAse) ', 'AGA_Bendability (DNAse) ',
           'AGC_Bendability (DNAse) ', 'AGG_Bendability (DNAse) ', 'AGT_Bendability (DNAse) ',
           'ATA_Bendability (DNAse) ', 'ATC_Bendability (DNAse) ', 'ATG_Bendability (DNAse) ',
           'ATT_Bendability (DNAse) ', 'CAA_Bendability (DNAse) ', 'CAC_Bendability (DNAse) ',
           'CAG_Bendability (DNAse) ', 'CAT_Bendability (DNAse) ', 'CCA_Bendability (DNAse) ',
           'CCC_Bendability (DNAse) ', 'CCG_Bendability (DNAse) ', 'CCT_Bendability (DNAse) ',
           'CGA_Bendability (DNAse) ', 'CGC_Bendability (DNAse) ', 'CGG_Bendability (DNAse) ',
           'CGT_Bendability (DNAse) ', 'CTA_Bendability (DNAse) ', 'CTC_Bendability (DNAse) ',
           'CTG_Bendability (DNAse) ', 'CTT_Bendability (DNAse) ', 'GAA_Bendability (DNAse) ',
           'GAC_Bendability (DNAse) ', 'GAG_Bendability (DNAse) ', 'GAT_Bendability (DNAse) ',
           'GCA_Bendability (DNAse) ', 'GCC_Bendability (DNAse) ', 'GCG_Bendability (DNAse) ',
           'GCT_Bendability (DNAse) ', 'GGA_Bendability (DNAse) ', 'GGC_Bendability (DNAse) ',
           'GGG_Bendability (DNAse) ', 'GGT_Bendability (DNAse) ', 'GTA_Bendability (DNAse) ',
           'GTC_Bendability (DNAse) ', 'GTG_Bendability (DNAse) ', 'GTT_Bendability (DNAse) ',
           'TAA_Bendability (DNAse) ', 'TAC_Bendability (DNAse) ', 'TAG_Bendability (DNAse) ',
           'TAT_Bendability (DNAse) ', 'TCA_Bendability (DNAse) ', 'TCC_Bendability (DNAse) ',
           'TCG_Bendability (DNAse) ', 'TCT_Bendability (DNAse) ', 'TGA_Bendability (DNAse) ',
           'TGC_Bendability (DNAse) ', 'TGG_Bendability (DNAse) ', 'TGT_Bendability (DNAse) ',
           'TTA_Bendability (DNAse) ', 'TTC_Bendability (DNAse) ', 'TTG_Bendability (DNAse) ',
           'TTT_Bendability (DNAse) ', 'AAA_Bendability (consensus) ', 'AAC_Bendability (consensus) ',
           'AAG_Bendability (consensus) ', 'AAT_Bendability (consensus) ', 'ACA_Bendability (consensus) ',
           'ACC_Bendability (consensus) ', 'ACG_Bendability (consensus) ', 'ACT_Bendability (consensus) ',
           'AGA_Bendability (consensus) ', 'AGC_Bendability (consensus) ', 'AGG_Bendability (consensus) ',
           'AGT_Bendability (consensus) ', 'ATA_Bendability (consensus) ', 'ATC_Bendability (consensus) ',
           'ATG_Bendability (consensus) ', 'ATT_Bendability (consensus) ', 'CAA_Bendability (consensus) ',
           'CAC_Bendability (consensus) ', 'CAG_Bendability (consensus) ', 'CAT_Bendability (consensus) ',
           'CCA_Bendability (consensus) ', 'CCC_Bendability (consensus) ', 'CCG_Bendability (consensus) ',
           'CCT_Bendability (consensus) ', 'CGA_Bendability (consensus) ', 'CGC_Bendability (consensus) ',
           'CGG_Bendability (consensus) ', 'CGT_Bendability (consensus) ', 'CTA_Bendability (consensus) ',
           'CTC_Bendability (consensus) ', 'CTG_Bendability (consensus) ', 'CTT_Bendability (consensus) ',
           'GAA_Bendability (consensus) ', 'GAC_Bendability (consensus) ', 'GAG_Bendability (consensus) ',
           'GAT_Bendability (consensus) ', 'GCA_Bendability (consensus) ', 'GCC_Bendability (consensus) ',
           'GCG_Bendability (consensus) ', 'GCT_Bendability (consensus) ', 'GGA_Bendability (consensus) ',
           'GGC_Bendability (consensus) ', 'GGG_Bendability (consensus) ', 'GGT_Bendability (consensus) ',
           'GTA_Bendability (consensus) ', 'GTC_Bendability (consensus) ', 'GTG_Bendability (consensus) ',
           'GTT_Bendability (consensus) ', 'TAA_Bendability (consensus) ', 'TAC_Bendability (consensus) ',
           'TAG_Bendability (consensus) ', 'TAT_Bendability (consensus) ', 'TCA_Bendability (consensus) ',
           'TCC_Bendability (consensus) ', 'TCG_Bendability (consensus) ', 'TCT_Bendability (consensus) ',
           'TGA_Bendability (consensus) ', 'TGC_Bendability (consensus) ', 'TGG_Bendability (consensus) ',
           'TGT_Bendability (consensus) ', 'TTA_Bendability (consensus) ', 'TTC_Bendability (consensus) ',
           'TTG_Bendability (consensus) ', 'TTT_Bendability (consensus) ', 'AAA_Trinucleotide  ',
           'AAC_Trinucleotide  ', 'AAG_Trinucleotide  ', 'AAT_Trinucleotide  ',
           'ACA_Trinucleotide  ', 'ACC_Trinucleotide  ', 'ACG_Trinucleotide  ',
           'ACT_Trinucleotide  ', 'AGA_Trinucleotide  ', 'AGC_Trinucleotide  ',
           'AGG_Trinucleotide  ', 'AGT_Trinucleotide  ', 'ATA_Trinucleotide  ',
           'ATC_Trinucleotide  ', 'ATG_Trinucleotide  ', 'ATT_Trinucleotide  ',
           'CAA_Trinucleotide  ', 'CAC_Trinucleotide  ', 'CAG_Trinucleotide  ',
           'CAT_Trinucleotide  ', 'CCA_Trinucleotide  ', 'CCC_Trinucleotide',
           'CCG_Trinucleotide  ', 'CCT_Trinucleotide  ', 'CGA_Trinucleotide  ',
           'CGC_Trinucleotide  ', 'CGG_Trinucleotide  ', 'CGT_Trinucleotide  ',
           'CTA_Trinucleotide  ', 'CTC_Trinucleotide  ', 'CTG_Trinucleotide  ',
           'CTT_Trinucleotide  ', 'GAA_Trinucleotide  ', 'GAC_Trinucleotide  ',
           'GAG_Trinucleotide  ', 'GAT_Trinucleotide  ', 'GCA_Trinucleotide  ',
           'GCC_Trinucleotide  ', 'GCG_Trinucleotide  ', 'GCT_Trinucleotide  ',
           'GGA_Trinucleotide  ', 'GGC_Trinucleotide  ', 'GGG_Trinucleotide  ',
           'GGT_Trinucleotide  ', 'GTA_Trinucleotide  ', 'GTC_Trinucleotide  ',
           'GTG_Trinucleotide  ', 'GTT_Trinucleotide  ', 'TAA_Trinucleotide  ',
           'TAC_Trinucleotide  ', 'TAG_Trinucleotide  ', 'TAT_Trinucleotide  ',
           'TCA_Trinucleotide  ', 'TCC_Trinucleotide  ', 'TCG_Trinucleotide  ',
           'TCT_Trinucleotide  ', 'TGA_Trinucleotide  ', 'TGC_Trinucleotide  ',
           'TGG_Trinucleotide  ', 'TGT_Trinucleotide  ', 'TTA_Trinucleotide  ',
           'TTC_Trinucleotide  ', 'TTG_Trinucleotide  ', 'TTT_Trinucleotide  ',
           'AAA_Nucleosome positioning ', 'AAC_Nucleosome positioning ', 'AAG_Nucleosome positioning ',
           'AAT_Nucleosome positioning ', 'ACA_Nucleosome positioning ', 'ACC_Nucleosome positioning ',
           'ACG_Nucleosome positioning ', 'ACT_Nucleosome positioning ', 'AGA_Nucleosome positioning ',
           'AGC_Nucleosome positioning ', 'AGG_Nucleosome positioning ', 'AGT_Nucleosome positioning ',
           'ATA_Nucleosome positioning ', 'ATC_Nucleosome positioning ', 'ATG_Nucleosome positioning ',
           'ATT_Nucleosome positioning ', 'CAA_Nucleosome positioning ', 'CAC_Nucleosome positioning ',
           'CAG_Nucleosome positioning ', 'CAT_Nucleosome positioning ', 'CCA_Nucleosome positioning ',
           'CCC_Nucleosome positioning ', 'CCG_Nucleosome positioning ', 'CCT_Nucleosome positioning ',
           'CGA_Nucleosome positioning ', 'CGC_Nucleosome positioning ', 'CGG_Nucleosome positioning ',
           'CGT_Nucleosome positioning ', 'CTA_Nucleosome positioning ', 'CTC_Nucleosome positioning ',
           'CTG_Nucleosome positioning ', 'CTT_Nucleosome positioning ', 'GAA_Nucleosome positioning ',
           'GAC_Nucleosome positioning ', 'GAG_Nucleosome positioning ', 'GAT_Nucleosome positioning ',
           'GCA_Nucleosome positioning ', 'GCC_Nucleosome positioning ', 'GCG_Nucleosome positioning ',
           'GCT_Nucleosome positioning ', 'GGA_Nucleosome positioning ', 'GGC_Nucleosome positioning ',
           'GGG_Nucleosome positioning ', 'GGT_Nucleosome positioning ', 'GTA_Nucleosome positioning ',
           'GTC_Nucleosome positioning ', 'GTG_Nucleosome positioning ', 'GTT_Nucleosome positioning ',
           'TAA_Nucleosome positioning ', 'TAC_Nucleosome positioning ', 'TAG_Nucleosome positioning ',
           'TAT_Nucleosome positioning ', 'TCA_Nucleosome positioning ', 'TCC_Nucleosome positioning ',
           'TCG_Nucleosome positioning ', 'TCT_Nucleosome positioning ', 'TGA_Nucleosome positioning ',
           'TGC_Nucleosome positioning ', 'TGG_Nucleosome positioning ', 'TGT_Nucleosome positioning ',
           'TTA_Nucleosome positioning ', 'TTC_Nucleosome positioning ', 'TTG_Nucleosome positioning ',
           'TTT_Nucleosome positioning ', 'AAA_Consensus_roll ', 'AAC_Consensus_roll ', 'AAG_Consensus_roll ',
           'AAT_Consensus_roll ', 'ACA_Consensus_roll ', 'ACC_Consensus_roll ', 'ACG_Consensus_roll ',
           'ACT_Consensus_roll ', 'AGA_Consensus_roll ', 'AGC_Consensus_roll ', 'AGG_Consensus_roll ',
           'AGT_Consensus_roll ', 'ATA_Consensus_roll ', 'ATC_Consensus_roll ', 'ATG_Consensus_roll ',
           'ATT_Consensus_roll ', 'CAA_Consensus_roll ', 'CAC_Consensus_roll ', 'CAG_Consensus_roll ',
           'CAT_Consensus_roll ', 'CCA_Consensus_roll ', 'CCC_Consensus_roll ', 'CCG_Consensus_roll ',
           'CCT_Consensus_roll ', 'CGA_Consensus_roll ', 'CGC_Consensus_roll ', 'CGG_Consensus_roll ',
           'CGT_Consensus_roll ', 'CTA_Consensus_roll ', 'CTC_Consensus_roll ', 'CTG_Consensus_roll ',
           'CTT_Consensus_roll ', 'GAA_Consensus_roll ', 'GAC_Consensus_roll ', 'GAG_Consensus_roll ',
           'GAT_Consensus_roll ', 'GCA_Consensus_roll ', 'GCC_Consensus_roll ', 'GCG_Consensus_roll ',
           'GCT_Consensus_roll ', 'GGA_Consensus_roll ', 'GGC_Consensus_roll ', 'GGG_Consensus_roll ',
           'GGT_Consensus_roll ', 'GTA_Consensus_roll ', 'GTC_Consensus_roll ', 'GTG_Consensus_roll ',
           'GTT_Consensus_roll ', 'TAA_Consensus_roll ', 'TAC_Consensus_roll ', 'TAG_Consensus_roll ',
           'TAT_Consensus_roll ', 'TCA_Consensus_roll ', 'TCC_Consensus_roll ', 'TCG_Consensus_roll ',
           'TCT_Consensus_roll ', 'TGA_Consensus_roll ', 'TGC_Consensus_roll ', 'TGG_Consensus_roll ',
           'TGT_Consensus_roll ', 'TTA_Consensus_roll ', 'TTC_Consensus_roll ', 'TTG_Consensus_roll ',
           'TTT_Consensus_roll ', 'AAA_Consensus-Rigid ', 'AAC_Consensus-Rigid ', 'AAG_Consensus-Rigid ',
           'AAT_Consensus-Rigid ', 'ACA_Consensus-Rigid ', 'ACC_Consensus-Rigid ', 'ACG_Consensus-Rigid ',
           'ACT_Consensus-Rigid ', 'AGA_Consensus-Rigid ', 'AGC_Consensus-Rigid ', 'AGG_Consensus-Rigid ',
           'AGT_Consensus-Rigid ', 'ATA_Consensus-Rigid ', 'ATC_Consensus-Rigid ', 'ATG_Consensus-Rigid ',
           'ATT_Consensus-Rigid ', 'CAA_Consensus-Rigid ', 'CAC_Consensus-Rigid ', 'CAG_Consensus-Rigid ',
           'CAT_Consensus-Rigid ', 'CCA_Consensus-Rigid ', 'CCC_Consensus-Rigid ', 'CCG_Consensus-Rigid ',
           'CCT_Consensus-Rigid ', 'CGA_Consensus-Rigid ', 'CGC_Consensus-Rigid ', 'CGG_Consensus-Rigid ',
           'CGT_Consensus-Rigid ', 'CTA_Consensus-Rigid ', 'CTC_Consensus-Rigid ', 'CTG_Consensus-Rigid ',
           'CTT_Consensus-Rigid ', 'GAA_Consensus-Rigid ', 'GAC_Consensus-Rigid ', 'GAG_Consensus-Rigid ',
           'GAT_Consensus-Rigid ', 'GCA_Consensus-Rigid ', 'GCC_Consensus-Rigid ', 'GCG_Consensus-Rigid ',
           'GCT_Consensus-Rigid ', 'GGA_Consensus-Rigid ', 'GGC_Consensus-Rigid ', 'GGG_Consensus-Rigid ',
           'GGT_Consensus-Rigid ', 'GTA_Consensus-Rigid ', 'GTC_Consensus-Rigid ', 'GTG_Consensus-Rigid ',
           'GTT_Consensus-Rigid ', 'TAA_Consensus-Rigid ', 'TAC_Consensus-Rigid ', 'TAG_Consensus-Rigid ',
           'TAT_Consensus-Rigid ', 'TCA_Consensus-Rigid ', 'TCC_Consensus-Rigid ', 'TCG_Consensus-Rigid ',
           'TCT_Consensus-Rigid ', 'TGA_Consensus-Rigid ', 'TGC_Consensus-Rigid ', 'TGG_Consensus-Rigid ',
           'TGT_Consensus-Rigid ', 'TTA_Consensus-Rigid ', 'TTC_Consensus-Rigid ', 'TTG_Consensus-Rigid ',
           'TTT_Consensus-Rigid ', 'AAA_Dnase I ', 'AAC_Dnase I ', 'AAG_Dnase I ', 'AAT_Dnase I ', 'ACA_Dnase I ',
           'ACC_Dnase I ', 'ACG_Dnase I ', 'ACT_Dnase I ', 'AGA_Dnase I ', 'AGC_Dnase I ', 'AGG_Dnase I ',
           'AGT_Dnase I ', 'ATA_Dnase I ', 'ATC_Dnase I ', 'ATG_Dnase I ', 'ATT_Dnase I ', 'CAA_Dnase I ',
           'CAC_Dnase I ', 'CAG_Dnase I ', 'CAT_Dnase I ', 'CCA_Dnase I ', 'CCC_Dnase I ', 'CCG_Dnase I ',
           'CCT_Dnase I ', 'CGA_Dnase I ', 'CGC_Dnase I ', 'CGG_Dnase I ', 'CGT_Dnase I ', 'CTA_Dnase I ',
           'CTC_Dnase I ', 'CTG_Dnase I ', 'CTT_Dnase I ', 'GAA_Dnase I ', 'GAC_Dnase I ', 'GAG_Dnase I ',
           'GAT_Dnase I ', 'GCA_Dnase I ', 'GCC_Dnase I ', 'GCG_Dnase I ', 'GCT_Dnase I ', 'GGA_Dnase I ',
           'GGC_Dnase I ', 'GGG_Dnase I ', 'GGT_Dnase I ', 'GTA_Dnase I ', 'GTC_Dnase I ', 'GTG_Dnase I ',
           'GTT_Dnase I ', 'TAA_Dnase I ', 'TAC_Dnase I ', 'TAG_Dnase I ', 'TAT_Dnase I ', 'TCA_Dnase I ',
           'TCC_Dnase I ', 'TCG_Dnase I ', 'TCT_Dnase I ', 'TGA_Dnase I ', 'TGC_Dnase I ', 'TGG_Dnase I ',
           'TGT_Dnase I ', 'TTA_Dnase I ', 'TTC_Dnase I ', 'TTG_Dnase I ', 'TTT_Dnase I ', 'AAA_Dnase I-Rigid ',
           'AAC_Dnase I-Rigid ', 'AAG_Dnase I-Rigid ', 'AAT_Dnase I-Rigid ', 'ACA_Dnase I-Rigid ', 'ACC_Dnase I-Rigid ',
           'ACG_Dnase I-Rigid ', 'ACT_Dnase I-Rigid ', 'AGA_Dnase I-Rigid ', 'AGC_Dnase I-Rigid ', 'AGG_Dnase I-Rigid ',
           'AGT_Dnase I-Rigid ', 'ATA_Dnase I-Rigid ', 'ATC_Dnase I-Rigid ', 'ATG_Dnase I-Rigid ', 'ATT_Dnase I-Rigid ',
           'CAA_Dnase I-Rigid ', 'CAC_Dnase I-Rigid ', 'CAG_Dnase I-Rigid ', 'CAT_Dnase I-Rigid ', 'CCA_Dnase I-Rigid ',
           'CCC_Dnase I-Rigid ', 'CCG_Dnase I-Rigid ', 'CCT_Dnase I-Rigid ', 'CGA_Dnase I-Rigid ', 'CGC_Dnase I-Rigid ',
           'CGG_Dnase I-Rigid ', 'CGT_Dnase I-Rigid ', 'CTA_Dnase I-Rigid ', 'CTC_Dnase I-Rigid ', 'CTG_Dnase I-Rigid ',
           'CTT_Dnase I-Rigid ', 'GAA_Dnase I-Rigid ', 'GAC_Dnase I-Rigid ', 'GAG_Dnase I-Rigid ', 'GAT_Dnase I-Rigid ',
           'GCA_Dnase I-Rigid ', 'GCC_Dnase I-Rigid ', 'GCG_Dnase I-Rigid ', 'GCT_Dnase I-Rigid ', 'GGA_Dnase I-Rigid ',
           'GGC_Dnase I-Rigid ', 'GGG_Dnase I-Rigid ', 'GGT_Dnase I-Rigid ', 'GTA_Dnase I-Rigid ', 'GTC_Dnase I-Rigid ',
           'GTG_Dnase I-Rigid ', 'GTT_Dnase I-Rigid ', 'TAA_Dnase I-Rigid ', 'TAC_Dnase I-Rigid ', 'TAG_Dnase I-Rigid ',
           'TAT_Dnase I-Rigid ', 'TCA_Dnase I-Rigid ', 'TCC_Dnase I-Rigid ', 'TCG_Dnase I-Rigid ', 'TCT_Dnase I-Rigid ',
           'TGA_Dnase I-Rigid ', 'TGC_Dnase I-Rigid ', 'TGG_Dnase I-Rigid ', 'TGT_Dnase I-Rigid ', 'TTA_Dnase I-Rigid ',
           'TTC_Dnase I-Rigid ', 'TTG_Dnase I-Rigid ', 'TTT_Dnase I-Rigid ', 'AAA_MW-Daltons ', 'AAC_MW-Daltons ',
           'AAG_MW-Daltons ', 'AAT_MW-Daltons ', 'ACA_MW-Daltons ', 'ACC_MW-Daltons ', 'ACG_MW-Daltons ',
           'ACT_MW-Daltons ', 'AGA_MW-Daltons ', 'AGC_MW-Daltons ', 'AGG_MW-Daltons ', 'AGT_MW-Daltons ',
           'ATA_MW-Daltons ', 'ATC_MW-Daltons ', 'ATG_MW-Daltons ', 'ATT_MW-Daltons ', 'CAA_MW-Daltons ',
           'CAC_MW-Daltons ', 'CAG_MW-Daltons ', 'CAT_MW-Daltons ', 'CCA_MW-Daltons ', 'CCC_MW-Daltons ',
           'CCG_MW-Daltons ', 'CCT_MW-Daltons ', 'CGA_MW-Daltons ', 'CGC_MW-Daltons ', 'CGG_MW-Daltons ',
           'CGT_MW-Daltons ', 'CTA_MW-Daltons ', 'CTC_MW-Daltons ', 'CTG_MW-Daltons ', 'CTT_MW-Daltons ',
           'GAA_MW-Daltons ', 'GAC_MW-Daltons ', 'GAG_MW-Daltons ', 'GAT_MW-Daltons ', 'GCA_MW-Daltons ',
           'GCC_MW-Daltons ', 'GCG_MW-Daltons ', 'GCT_MW-Daltons ', 'GGA_MW-Daltons ', 'GGC_MW-Daltons ',
           'GGG_MW-Daltons ', 'GGT_MW-Daltons ', 'GTA_MW-Daltons ', 'GTC_MW-Daltons ', 'GTG_MW-Daltons ',
           'GTT_MW-Daltons ', 'TAA_MW-Daltons ', 'TAC_MW-Daltons ', 'TAG_MW-Daltons ', 'TAT_MW-Daltons ',
           'TCA_MW-Daltons ', 'TCC_MW-Daltons ', 'TCG_MW-Daltons ', 'TCT_MW-Daltons ', 'TGA_MW-Daltons ',
           'TGC_MW-Daltons ', 'TGG_MW-Daltons ', 'TGT_MW-Daltons ', 'TTA_MW-Daltons ', 'TTC_MW-Daltons ',
           'TTG_MW-Daltons ', 'TTT_MW-Daltons ', 'AAA_MW-kg ', 'AAC_MW-kg ', 'AAG_MW-kg ', 'AAT_MW-kg ', 'ACA_MW-kg ',
           'ACC_MW-kg ', 'ACG_MW-kg ', 'ACT_MW-kg ', 'AGA_MW-kg ', 'AGC_MW-kg ', 'AGG_MW-kg ', 'AGT_MW-kg ',
           'ATA_MW-kg ', 'ATC_MW-kg ', 'ATG_MW-kg ', 'ATT_MW-kg ', 'CAA_MW-kg ', 'CAC_MW-kg ', 'CAG_MW-kg ',
           'CAT_MW-kg ', 'CCA_MW-kg ', 'CCC_MW-kg ', 'CCG_MW-kg ', 'CCT_MW-kg ', 'CGA_MW-kg ', 'CGC_MW-kg ',
           'CGG_MW-kg ', 'CGT_MW-kg ', 'CTA_MW-kg ', 'CTC_MW-kg ', 'CTG_MW-kg ', 'CTT_MW-kg ', 'GAA_MW-kg ',
           'GAC_MW-kg ', 'GAG_MW-kg ', 'GAT_MW-kg ', 'GCA_MW-kg ', 'GCC_MW-kg ', 'GCG_MW-kg ', 'GCT_MW-kg ',
           'GGA_MW-kg ', 'GGC_MW-kg ', 'GGG_MW-kg ', 'GGT_MW-kg ', 'GTA_MW-kg ', 'GTC_MW-kg ', 'GTG_MW-kg ',
           'GTT_MW-kg ', 'TAA_MW-kg ', 'TAC_MW-kg ', 'TAG_MW-kg ', 'TAT_MW-kg ', 'TCA_MW-kg ', 'TCC_MW-kg ',
           'TCG_MW-kg ', 'TCT_MW-kg ', 'TGA_MW-kg ', 'TGC_MW-kg ', 'TGG_MW-kg ', 'TGT_MW-kg ', 'TTA_MW-kg ',
           'TTC_MW-kg ', 'TTG_MW-kg ', 'TTT_MW-kg ', 'AAA_Nucleosome ', 'AAC_Nucleosome ', 'AAG_Nucleosome ',
           'AAT_Nucleosome ', 'ACA_Nucleosome ', 'ACC_Nucleosome ', 'ACG_Nucleosome ', 'ACT_Nucleosome ',
           'AGA_Nucleosome ', 'AGC_Nucleosome ', 'AGG_Nucleosome ', 'AGT_Nucleosome ', 'ATA_Nucleosome ',
           'ATC_Nucleosome ', 'ATG_Nucleosome ', 'ATT_Nucleosome ', 'CAA_Nucleosome ', 'CAC_Nucleosome ',
           'CAG_Nucleosome ', 'CAT_Nucleosome ', 'CCA_Nucleosome ', 'CCC_Nucleosome ', 'CCG_Nucleosome ',
           'CCT_Nucleosome ', 'CGA_Nucleosome ', 'CGC_Nucleosome ', 'CGG_Nucleosome ', 'CGT_Nucleosome ',
           'CTA_Nucleosome ', 'CTC_Nucleosome ', 'CTG_Nucleosome ', 'CTT_Nucleosome ', 'GAA_Nucleosome ',
           'GAC_Nucleosome ', 'GAG_Nucleosome ', 'GAT_Nucleosome ', 'GCA_Nucleosome ', 'GCC_Nucleosome ',
           'GCG_Nucleosome ', 'GCT_Nucleosome ', 'GGA_Nucleosome ', 'GGC_Nucleosome ', 'GGG_Nucleosome ',
           'GGT_Nucleosome ', 'GTA_Nucleosome ', 'GTC_Nucleosome ', 'GTG_Nucleosome ', 'GTT_Nucleosome ',
           'TAA_Nucleosome ', 'TAC_Nucleosome ', 'TAG_Nucleosome ', 'TAT_Nucleosome ', 'TCA_Nucleosome ',
           'TCC_Nucleosome ', 'TCG_Nucleosome ', 'TCT_Nucleosome ', 'TGA_Nucleosome ', 'TGC_Nucleosome ',
           'TGG_Nucleosome ', 'TGT_Nucleosome ', 'TTA_Nucleosome ', 'TTC_Nucleosome ', 'TTG_Nucleosome ',
           'TTT_Nucleosome ', 'AAA_Nucleosome-Rigid ', 'AAC_Nucleosome-Rigid ', 'AAG_Nucleosome-Rigid ',
           'AAT_Nucleosome-Rigid ', 'ACA_Nucleosome-Rigid ', 'ACC_Nucleosome-Rigid ', 'ACG_Nucleosome-Rigid ',
           'ACT_Nucleosome-Rigid ', 'AGA_Nucleosome-Rigid ', 'AGC_Nucleosome-Rigid ', 'AGG_Nucleosome-Rigid ',
           'AGT_Nucleosome-Rigid ', 'ATA_Nucleosome-Rigid ', 'ATC_Nucleosome-Rigid ', 'ATG_Nucleosome-Rigid ',
           'ATT_Nucleosome-Rigid ', 'CAA_Nucleosome-Rigid ', 'CAC_Nucleosome-Rigid ', 'CAG_Nucleosome-Rigid ',
           'CAT_Nucleosome-Rigid ', 'CCA_Nucleosome-Rigid ', 'CCC_Nucleosome-Rigid ', 'CCG_Nucleosome-Rigid ',
           'CCT_Nucleosome-Rigid ', 'CGA_Nucleosome-Rigid ', 'CGC_Nucleosome-Rigid ', 'CGG_Nucleosome-Rigid ',
           'CGT_Nucleosome-Rigid ', 'CTA_Nucleosome-Rigid ', 'CTC_Nucleosome-Rigid ', 'CTG_Nucleosome-Rigid ',
           'CTT_Nucleosome-Rigid ', 'GAA_Nucleosome-Rigid ', 'GAC_Nucleosome-Rigid ', 'GAG_Nucleosome-Rigid ',
           'GAT_Nucleosome-Rigid ', 'GCA_Nucleosome-Rigid ', 'GCC_Nucleosome-Rigid ', 'GCG_Nucleosome-Rigid ',
           'GCT_Nucleosome-Rigid ', 'GGA_Nucleosome-Rigid ', 'GGC_Nucleosome-Rigid ', 'GGG_Nucleosome-Rigid ',
           'GGT_Nucleosome-Rigid ', 'GTA_Nucleosome-Rigid ', 'GTC_Nucleosome-Rigid ', 'GTG_Nucleosome-Rigid ',
           'GTT_Nucleosome-Rigid ', 'TAA_Nucleosome-Rigid ', 'TAC_Nucleosome-Rigid ', 'TAG_Nucleosome-Rigid ',
           'TAT_Nucleosome-Rigid ', 'TCA_Nucleosome-Rigid ', 'TCC_Nucleosome-Rigid ', 'TCG_Nucleosome-Rigid ',
           'TCT_Nucleosome-Rigid ', 'TGA_Nucleosome-Rigid ', 'TGC_Nucleosome-Rigid ', 'TGG_Nucleosome-Rigid ',
           'TGT_Nucleosome-Rigid ', 'TTA_Nucleosome-Rigid ', 'TTC_Nucleosome-Rigid ', 'TTG_Nucleosome-Rigid ',
           'TTT_Nucleosome-Rigid ', 'Rise.lag1 ', 'Rise.lag2 ', 'Rise.lag3 ', 'Rise.lag4 ', 'Rise.lag5 ', 'Rise.lag6 ',
           'Rise.lag7 ', 'Rise.lag8 ', 'Rise.lag9 ', 'Rise.lag10 ', 'Roll.lag1 ', 'Roll.lag2 ', 'Roll.lag3 ',
           'Roll.lag4 ', 'Roll.lag5 ', 'Roll.lag6 ', 'Roll.lag7 ', 'Roll.lag8 ', 'Roll.lag9 ', 'Roll.lag10 ',
           'Shift.lag1 ', 'Shift.lag2 ', 'Shift.lag3 ', 'Shift.lag4 ', 'Shift.lag5 ', 'Shift.lag6 ', 'Shift.lag7 ',
           'Shift.lag8 ', 'Shift.lag9 ', 'Shift.lag10 ', 'Slide.lag1 ', 'Slide.lag2 ', 'Slide.lag3 ', 'Slide.lag4 ',
           'Slide.lag5 ', 'Slide.lag6 ', 'Slide.lag7 ', 'Slide.lag8 ', 'Slide.lag9 ', 'Slide.lag10 ', 'Tilt.lag1 ',
           'Tilt.lag2 ', 'Tilt.lag3 ', 'Tilt.lag4 ', 'Tilt.lag5 ', 'Tilt.lag6 ', 'Tilt.lag7 ', 'Tilt.lag8 ',
           'Tilt.lag9 ', 'Tilt.lag10 ', 'Twist.lag1 ', 'Twist.lag2 ', 'Twist.lag3 ', 'Twist.lag4 ', 'Twist.lag5 ',
           'Twist.lag6 ', 'Twist.lag7 ', 'Twist.lag8 ', 'Twist.lag9 ', 'Twist.lag10 ', 'Rise-Roll-lag.1 ',
           'Rise-Roll-lag.2 ', 'Rise-Roll-lag.3 ', 'Rise-Roll-lag.4 ', 'Rise-Roll-lag.5 ', 'Rise-Roll-lag.6 ',
           'Rise-Roll-lag.7 ', 'Rise-Roll-lag.8 ', 'Rise-Roll-lag.9 ', 'Rise-Roll-lag.10 ', 'Roll-Rise-lag.1 ',
           'Roll-Rise-lag.2 ', 'Roll-Rise-lag.3 ', 'Roll-Rise-lag.4 ', 'Roll-Rise-lag.5 ', 'Roll-Rise-lag.6 ',
           'Roll-Rise-lag.7 ', 'Roll-Rise-lag.8 ', 'Roll-Rise-lag.9 ', 'Roll-Rise-lag.10 ', 'Rise-Shift-lag.1 ',
           'Rise-Shift-lag.2 ', 'Rise-Shift-lag.3 ', 'Rise-Shift-lag.4 ', 'Rise-Shift-lag.5 ', 'Rise-Shift-lag.6 ',
           'Rise-Shift-lag.7 ', 'Rise-Shift-lag.8 ', 'Rise-Shift-lag.9 ', 'Rise-Shift-lag.10 ', 'Shift-Rise-lag.1 ',
           'Shift-Rise-lag.2 ', 'Shift-Rise-lag.3 ', 'Shift-Rise-lag.4 ', 'Shift-Rise-lag.5 ', 'Shift-Rise-lag.6 ',
           'Shift-Rise-lag.7 ', 'Shift-Rise-lag.8 ', 'Shift-Rise-lag.9 ', 'Shift-Rise-lag.10 ', 'Rise-Slide-lag.1 ',
           'Rise-Slide-lag.2 ', 'Rise-Slide-lag.3 ', 'Rise-Slide-lag.4 ', 'Rise-Slide-lag.5 ', 'Rise-Slide-lag.6 ',
           'Rise-Slide-lag.7 ', 'Rise-Slide-lag.8 ', 'Rise-Slide-lag.9 ', 'Rise-Slide-lag.10 ', 'Slide-Rise-lag.1 ',
           'Slide-Rise-lag.2 ', 'Slide-Rise-lag.3 ', 'Slide-Rise-lag.4 ', 'Slide-Rise-lag.5 ', 'Slide-Rise-lag.6 ',
           'Slide-Rise-lag.7 ', 'Slide-Rise-lag.8 ', 'Slide-Rise-lag.9 ', 'Slide-Rise-lag.10 ', 'Rise-Tilt-lag.1 ',
           'Rise-Tilt-lag.2 ', 'Rise-Tilt-lag.3 ', 'Rise-Tilt-lag.4 ', 'Rise-Tilt-lag.5 ', 'Rise-Tilt-lag.6 ',
           'Rise-Tilt-lag.7 ', 'Rise-Tilt-lag.8 ', 'Rise-Tilt-lag.9 ', 'Rise-Tilt-lag.10 ', 'Tilt-Rise-lag.1 ',
           'Tilt-Rise-lag.2 ', 'Tilt-Rise-lag.3 ', 'Tilt-Rise-lag.4 ', 'Tilt-Rise-lag.5 ', 'Tilt-Rise-lag.6 ',
           'Tilt-Rise-lag.7 ', 'Tilt-Rise-lag.8 ', 'Tilt-Rise-lag.9 ', 'Tilt-Rise-lag.10 ', 'Rise-Twist-lag.1 ',
           'Rise-Twist-lag.2 ', 'Rise-Twist-lag.3 ', 'Rise-Twist-lag.4 ', 'Rise-Twist-lag.5 ', 'Rise-Twist-lag.6 ',
           'Rise-Twist-lag.7 ', 'Rise-Twist-lag.8 ', 'Rise-Twist-lag.9 ', 'Rise-Twist-lag.10 ', 'Twist-Rise-lag.1 ',
           'Twist-Rise-lag.2 ', 'Twist-Rise-lag.3 ', 'Twist-Rise-lag.4 ', 'Twist-Rise-lag.5 ', 'Twist-Rise-lag.6 ',
           'Twist-Rise-lag.7 ', 'Twist-Rise-lag.8 ', 'Twist-Rise-lag.9 ', 'Twist-Rise-lag.10 ', 'Roll-Shift-lag.1 ',
           'Roll-Shift-lag.2 ', 'Roll-Shift-lag.3 ', 'Roll-Shift-lag.4 ', 'Roll-Shift-lag.5 ', 'Roll-Shift-lag.6 ',
           'Roll-Shift-lag.7 ', 'Roll-Shift-lag.8 ', 'Roll-Shift-lag.9 ', 'Roll-Shift-lag.10 ', 'Shift-Roll-lag.1 ',
           'Shift-Roll-lag.2 ', 'Shift-Roll-lag.3 ', 'Shift-Roll-lag.4 ', 'Shift-Roll-lag.5 ', 'Shift-Roll-lag.6 ',
           'Shift-Roll-lag.7 ', 'Shift-Roll-lag.8 ', 'Shift-Roll-lag.9 ', 'Shift-Roll-lag.10 ', 'Roll-Slide-lag.1 ',
           'Roll-Slide-lag.2 ', 'Roll-Slide-lag.3 ', 'Roll-Slide-lag.4 ', 'Roll-Slide-lag.5 ', 'Roll-Slide-lag.6 ',
           'Roll-Slide-lag.7 ', 'Roll-Slide-lag.8 ', 'Roll-Slide-lag.9 ', 'Roll-Slide-lag.10 ', 'Slide-Roll-lag.1 ',
           'Slide-Roll-lag.2 ', 'Slide-Roll-lag.3 ', 'Slide-Roll-lag.4 ', 'Slide-Roll-lag.5 ', 'Slide-Roll-lag.6 ',
           'Slide-Roll-lag.7 ', 'Slide-Roll-lag.8 ', 'Slide-Roll-lag.9 ', 'Slide-Roll-lag.10 ', 'Roll-Tilt-lag.1 ',
           'Roll-Tilt-lag.2 ', 'Roll-Tilt-lag.3 ', 'Roll-Tilt-lag.4 ', 'Roll-Tilt-lag.5 ', 'Roll-Tilt-lag.6 ',
           'Roll-Tilt-lag.7 ', 'Roll-Tilt-lag.8 ', 'Roll-Tilt-lag.9 ', 'Roll-Tilt-lag.10 ', 'Tilt-Roll-lag.1 ',
           'Tilt-Roll-lag.2 ', 'Tilt-Roll-lag.3 ', 'Tilt-Roll-lag.4 ', 'Tilt-Roll-lag.5 ', 'Tilt-Roll-lag.6 ',
           'Tilt-Roll-lag.7 ', 'Tilt-Roll-lag.8 ', 'Tilt-Roll-lag.9 ', 'Tilt-Roll-lag.10 ', 'Roll-Twist-lag.1 ',
           'Roll-Twist-lag.2 ', 'Roll-Twist-lag.3 ', 'Roll-Twist-lag.4 ', 'Roll-Twist-lag.5 ', 'Roll-Twist-lag.6 ',
           'Roll-Twist-lag.7 ', 'Roll-Twist-lag.8 ', 'Roll-Twist-lag.9 ', 'Roll-Twist-lag.10 ', 'Twist-Roll-lag.1 ',
           'Twist-Roll-lag.2 ', 'Twist-Roll-lag.3 ', 'Twist-Roll-lag.4 ', 'Twist-Roll-lag.5 ', 'Twist-Roll-lag.6 ',
           'Twist-Roll-lag.7 ', 'Twist-Roll-lag.8 ', 'Twist-Roll-lag.9 ', 'Twist-Roll-lag.10 ', 'Shift-Slide-lag.1 ',
           'Shift-Slide-lag.2 ', 'Shift-Slide-lag.3 ', 'Shift-Slide-lag.4 ', 'Shift-Slide-lag.5 ', 'Shift-Slide-lag.6 ',
           'Shift-Slide-lag.7 ', 'Shift-Slide-lag.8 ', 'Shift-Slide-lag.9 ', 'Shift-Slide-lag.10 ',
           'Slide-Shift-lag.1 ', 'Slide-Shift-lag.2 ', 'Slide-Shift-lag.3 ', 'Slide-Shift-lag.4 ', 'Slide-Shift-lag.5 ',
           'Slide-Shift-lag.6 ', 'Slide-Shift-lag.7 ', 'Slide-Shift-lag.8 ', 'Slide-Shift-lag.9 ',
           'Slide-Shift-lag.10 ', 'Shift-Tilt-lag.1 ', 'Shift-Tilt-lag.2 ', 'Shift-Tilt-lag.3 ', 'Shift-Tilt-lag.4 ',
           'Shift-Tilt-lag.5 ', 'Shift-Tilt-lag.6 ', 'Shift-Tilt-lag.7 ', 'Shift-Tilt-lag.8 ', 'Shift-Tilt-lag.9 ',
           'Shift-Tilt-lag.10 ', 'Tilt-Shift-lag.1 ', 'Tilt-Shift-lag.2 ', 'Tilt-Shift-lag.3 ', 'Tilt-Shift-lag.4 ',
           'Tilt-Shift-lag.5 ', 'Tilt-Shift-lag.6 ', 'Tilt-Shift-lag.7 ', 'Tilt-Shift-lag.8 ', 'Tilt-Shift-lag.9 ',
           'Tilt-Shift-lag.10 ', 'Shift-Twist-lag.1 ', 'Shift-Twist-lag.2 ', 'Shift-Twist-lag.3 ', 'Shift-Twist-lag.4 ',
           'Shift-Twist-lag.5 ', 'Shift-Twist-lag.6 ', 'Shift-Twist-lag.7 ', 'Shift-Twist-lag.8 ', 'Shift-Twist-lag.9 ',
           'Shift-Twist-lag.10 ', 'Twist-Shift-lag.1 ', 'Twist-Shift-lag.2 ', 'Twist-Shift-lag.3 ',
           'Twist-Shift-lag.4 ', 'Twist-Shift-lag.5 ', 'Twist-Shift-lag.6 ', 'Twist-Shift-lag.7 ', 'Twist-Shift-lag.8 ',
           'Twist-Shift-lag.9 ', 'Twist-Shift-lag.10 ', 'Slide-Tilt-lag.1 ', 'Slide-Tilt-lag.2 ', 'Slide-Tilt-lag.3 ',
           'Slide-Tilt-lag.4 ', 'Slide-Tilt-lag.5 ', 'Slide-Tilt-lag.6 ', 'Slide-Tilt-lag.7 ', 'Slide-Tilt-lag.8 ',
           'Slide-Tilt-lag.9 ', 'Slide-Tilt-lag.10 ', 'Tilt-Slide-lag.1 ', 'Tilt-Slide-lag.2 ', 'Tilt-Slide-lag.3 ',
           'Tilt-Slide-lag.4 ', 'Tilt-Slide-lag.5 ', 'Tilt-Slide-lag.6 ', 'Tilt-Slide-lag.7 ', 'Tilt-Slide-lag.8 ',
           'Tilt-Slide-lag.9 ', 'Tilt-Slide-lag.10 ', 'Slide-Twist-lag.1 ', 'Slide-Twist-lag.2 ', 'Slide-Twist-lag.3 ',
           'Slide-Twist-lag.4 ', 'Slide-Twist-lag.5 ', 'Slide-Twist-lag.6 ', 'Slide-Twist-lag.7 ', 'Slide-Twist-lag.8 ',
           'Slide-Twist-lag.9 ', 'Slide-Twist-lag.10 ', 'Twist-Slide-lag.1 ', 'Twist-Slide-lag.2 ',
           'Twist-Slide-lag.3 ', 'Twist-Slide-lag.4 ', 'Twist-Slide-lag.5 ', 'Twist-Slide-lag.6 ', 'Twist-Slide-lag.7 ',
           'Twist-Slide-lag.8 ', 'Twist-Slide-lag.9 ', 'Twist-Slide-lag.10 ', 'Tilt-Twist-lag.1 ', 'Tilt-Twist-lag.2 ',
           'Tilt-Twist-lag.3 ', 'Tilt-Twist-lag.4 ', 'Tilt-Twist-lag.5 ', 'Tilt-Twist-lag.6 ', 'Tilt-Twist-lag.7 ',
           'Tilt-Twist-lag.8 ', 'Tilt-Twist-lag.9 ', 'Tilt-Twist-lag.10 ', 'Twist-Tilt-lag.1 ', 'Twist-Tilt-lag.2 ',
           'Twist-Tilt-lag.3 ', 'Twist-Tilt-lag.4 ', 'Twist-Tilt-lag.5 ', 'Twist-Tilt-lag.6 ', 'Twist-Tilt-lag.7 ',
           'Twist-Tilt-lag.8 ', 'Twist-Tilt-lag.9 ', 'Twist-Tilt-lag.10 ', 'MMI_AA ', 'MMI_AC ', 'MMI_AG ', 'MMI_AT ',
           'MMI_CC ', 'MMI_CG ', 'MMI_CT ', 'MMI_GG ', 'MMI_GT ', 'MMI_TT ', 'MMI_AAA ', 'MMI_AAC ', 'MMI_AAG ',
           'MMI_AAT ', 'MMI_ACC ', 'MMI_ACG ', 'MMI_ACT ', 'MMI_AGG ', 'MMI_AGT ', 'MMI_ATT ', 'MMI_CCC ', 'MMI_CCG ',
           'MMI_CCT ', 'MMI_CGG ', 'MMI_CGT ', 'MMI_CTT ', 'MMI_GGG ', 'MMI_GGT ', 'MMI_GTT ', 'MMI_TTT ',
           'Base stacking.lag1 ', 'Base stacking.lag2 ', 'Protein induced deformability.lag1 ',
           'Protein induced deformability.lag2 ', 'B-DNA twist.lag1 ', 'B-DNA twist.lag2 ',
           'Dinucleotide .lag1 ', 'Dinucleotide .lag2 ', 'A-philicity.lag1 ', 'A-philicity.lag2 ',
           'Propeller twist.lag1 ', 'Propeller twist.lag2 ', 'Duplex stability:(freeenergy).lag1 ',
           'Duplex stability:(freeenergy).lag2 ', 'Duplex tability(disruptenergy).lag1 ',
           'Duplex tability(disruptenergy).lag2 ', 'DNA denaturation.lag1 ', 'DNA denaturation.lag2 ',
           'Bending stiffness.lag1 ', 'Bending stiffness.lag2 ', 'Protein DNA twist.lag1 ', 'Protein DNA twist.lag2 ',
           'Stabilising energy of Z-DNA.lag1 ', 'Stabilising energy of Z-DNA.lag2 ', 'Aida_BA_transition.lag1 ',
           'Aida_BA_transition.lag2 ', 'Breslauer_dG.lag1 ', 'Breslauer_dG.lag2 ', 'Breslauer_dH.lag1 ',
           'Breslauer_dH.lag2 ', 'Breslauer_dS.lag1 ', 'Breslauer_dS.lag2 ', 'Electron_interaction.lag1 ',
           'Electron_interaction.lag2 ', 'Hartman_trans_free_energy.lag1 ', 'Hartman_trans_free_energy.lag2 ',
           'Helix-Coil_transition.lag1 ', 'Helix-Coil_transition.lag2 ', 'Ivanov_BA_transition.lag1 ',
           'Ivanov_BA_transition.lag2 ', 'Lisser_BZ_transition.lag1 ', 'Lisser_BZ_transition.lag2 ',
           'Polar_interaction.lag1 ', 'Polar_interaction.lag2 ', 'SantaLucia_dG.lag1 ', 'SantaLucia_dG.lag2 ',
           'SantaLucia_dH.lag1 ', 'SantaLucia_dH.lag2 ', 'SantaLucia_dS.lag1 ', 'SantaLucia_dS.lag2 ',
           'Sarai_flexibility.lag1 ', 'Sarai_flexibility.lag2 ', 'Stability.lag1 ', 'Stability.lag2 ',
           'Stacking_energy.lag1 ', 'Stacking_energy.lag2 ', 'Sugimoto_dG.lag1 ', 'Sugimoto_dG.lag2 ',
           'Sugimoto_dH.lag1 ', 'Sugimoto_dH.lag2 ', 'Sugimoto_dS.lag1 ', 'Sugimoto_dS.lag2 ',
           'Watson-Crick_interaction.lag1 ', 'Watson-Crick_interaction.lag2 ', 'Twist.lag1 ', 'Twist.lag2 ',
           'Tilt.lag1 ', 'Tilt.lag2 ', 'Roll.lag1 ', 'Roll.lag2 ', 'Shift.lag1 ', 'Shift.lag2 ', 'Slide.lag1 ',
           'Slide.lag2 ', 'Rise.lag1 ', 'Rise.lag2 ', 'Clash Strength.lag1 ', 'Clash Strength.lag2 ', 'Roll_roll.lag1 ',
           'Roll_roll.lag2 ', 'Twist stiffness.lag1 ', 'Twist stiffness.lag2 ', 'Tilt stiffness.lag1 ',
           'Tilt stiffness.lag2 ', 'Shift_rise.lag1 ', 'Shift_rise.lag2 ', 'Adenine content.lag1 ',
           'Adenine content.lag2 ', 'Direction.lag1 ', 'Direction.lag2 ', 'Twist_shift.lag1 ', 'Twist_shift.lag2 ',
           'Enthalpy1.lag1 ', 'Enthalpy1.lag2 ', 'Twist_twist.lag1 ', 'Twist_twist.lag2 ', 'Roll_shift.lag1 ',
           'Roll_shift.lag2 ', 'Shift_slide.lag1 ', 'Shift_slide.lag2 ', 'Shift2.lag1 ', 'Shift2.lag2 ', 'Tilt3.lag1 ',
           'Tilt3.lag2 ', 'Tilt1.lag1 ', 'Tilt1.lag2 ', 'Tilt4.lag1 ', 'Tilt4.lag2 ', 'Tilt2.lag1 ', 'Tilt2.lag2 ',
           'Slide (DNA-protein complex)1.lag1 ', 'Slide (DNA-protein complex)1.lag2 ', 'Tilt_shift.lag1 ',
           'Tilt_shift.lag2 ', 'Twist_tilt.lag1 ', 'Twist_tilt.lag2 ', 'Twist (DNA-protein complex)1.lag1 ',
           'Twist (DNA-protein complex)1.lag2 ', 'Tilt_rise.lag1 ', 'Tilt_rise.lag2 ', 'Roll_rise.lag1 ',
           'Roll_rise.lag2 ', 'Stacking energy.lag1 ', 'Stacking energy.lag2 ', 'Stacking energy1.lag1 ',
           'Stacking energy1.lag2 ', 'Stacking energy2.lag1 ', 'Stacking energy2.lag2 ', 'Stacking energy3.lag1 ',
           'Stacking energy3.lag2 ', 'Propeller Twist.lag1 ', 'Propeller Twist.lag2 ', 'Roll11.lag1 ', 'Roll11.lag2 ',
           'Rise (DNA-protein complex).lag1 ', 'Rise (DNA-protein complex).lag2 ', 'Tilt_tilt.lag1 ', 'Tilt_tilt.lag2 ',
           'Roll4.lag1 ', 'Roll4.lag2 ', 'Roll2.lag1 ', 'Roll2.lag2 ', 'Roll3.lag1 ', 'Roll3.lag2 ', 'Roll1.lag1 ',
           'Roll1.lag2 ', 'Minor Groove Size.lag1 ', 'Minor Groove Size.lag2 ', '.lag1 ', '.lag2 ',
           'Slide_slide.lag1 ', 'Slide_slide.lag2 ', 'Enthalpy.lag1 ', 'Enthalpy.lag2 ', 'Shift_shift.lag1 ',
           'Shift_shift.lag2 ', 'Slide stiffness.lag1 ', 'Slide stiffness.lag2 ', 'Melting Temperature1.lag1 ',
           'Melting Temperature1.lag2 ', 'Flexibility_slide.lag1 ', 'Flexibility_slide.lag2 ',
           'Minor Groove Distance.lag1 ', 'Minor Groove Distance.lag2 ', 'Rise (DNA-protein complex)1.lag1 ',
           'Rise (DNA-protein complex)1.lag2 ', 'Tilt (DNA-protein complex).lag1 ', 'Tilt (DNA-protein complex).lag2 ',
           'Guanine content.lag1 ', 'Guanine content.lag2 ', 'Roll (DNA-protein complex)1.lag1 ',
           'Roll (DNA-protein complex)1.lag2 ', 'Entropy.lag1 ', 'Entropy.lag2 ', 'Cytosine content.lag1 ',
           'Cytosine content.lag2 ', 'Major Groove Size.lag1 ', 'Major Groove Size.lag2 ', 'Twist_rise.lag1 ',
           'Twist_rise.lag2 ', 'Major Groove Distance.lag1 ', 'Major Groove Distance.lag2 ',
           'Twist (DNA-protein complex).lag1 ', 'Twist (DNA-protein complex).lag2 ', 'Purine (AG) content.lag1 ',
           'Purine (AG) content.lag2 ', 'Melting Temperature.lag1 ', 'Melting Temperature.lag2 ', 'Free energy.lag1 ',
           'Free energy.lag2 ', 'Tilt_slide.lag1 ', 'Tilt_slide.lag2 ', 'Major Groove Width.lag1 ',
           'Major Groove Width.lag2 ', 'Major Groove Depth.lag1 ', 'Major Groove Depth.lag2 ', 'Wedge.lag1 ',
           'Wedge.lag2 ', 'Free energy8.lag1 ', 'Free energy8.lag2 ', 'Free energy6.lag1 ', 'Free energy6.lag2 ',
           'Free energy7.lag1 ', 'Free energy7.lag2 ', 'Free energy4.lag1 ', 'Free energy4.lag2 ', 'Free energy5.lag1 ',
           'Free energy5.lag2 ', 'Free energy2.lag1 ', 'Free energy2.lag2 ', 'Free energy3.lag1 ', 'Free energy3.lag2 ',
           'Free energy1.lag1 ', 'Free energy1.lag2 ', 'Twist_roll.lag1 ', 'Twist_roll.lag2 ',
           'Shift (DNA-protein complex).lag1 ', 'Shift (DNA-protein complex).lag2 ', 'Rise_rise.lag1 ',
           'Rise_rise.lag2 ', 'Flexibility_shift.lag1 ', 'Flexibility_shift.lag2 ',
           'Shift (DNA-protein complex)1.lag1 ', 'Shift (DNA-protein complex)1.lag2 ', 'Thymine content.lag1 ',
           'Thymine content.lag2 ', 'Slide_rise.lag1 ', 'Slide_rise.lag2 ', 'Tilt_roll.lag1 ', 'Tilt_roll.lag2 ',
           'Tip.lag1 ', 'Tip.lag2 ', 'Keto (GT) content.lag1 ', 'Keto (GT) content.lag2 ', 'Roll stiffness.lag1 ',
           'Roll stiffness.lag2 ', 'Minor Groove Width.lag1 ', 'Minor Groove Width.lag2 ', 'Inclination.lag1 ',
           'Inclination.lag2 ', 'Entropy1.lag1 ', 'Entropy1.lag2 ', 'Roll_slide.lag1 ', 'Roll_slide.lag2 ',
           'Slide (DNA-protein complex).lag1 ', 'Slide (DNA-protein complex).lag2 ', 'Twist1.lag1 ', 'Twist1.lag2 ',
           'Twist3.lag1 ', 'Twist3.lag2 ', 'Twist2.lag1 ', 'Twist2.lag2 ', 'Twist5.lag1 ', 'Twist5.lag2 ',
           'Twist4.lag1 ', 'Twist4.lag2 ', 'Twist7.lag1 ', 'Twist7.lag2 ', 'Twist6.lag1 ', 'Twist6.lag2 ',
           'Tilt (DNA-protein complex)1.lag1 ', 'Tilt (DNA-protein complex)1.lag2 ', 'Twist_slide.lag1 ',
           'Twist_slide.lag2 ', 'Minor Groove Depth.lag1 ', 'Minor Groove Depth.lag2 ',
           'Roll (DNA-protein complex).lag1 ', 'Roll (DNA-protein complex).lag2 ', 'Rise2.lag1 ', 'Rise2.lag2 ',
           'Persistance Length.lag1 ', 'Persistance Length.lag2 ', 'Rise3.lag1 ', 'Rise3.lag2 ',
           'Shift stiffness.lag1 ', 'Shift stiffness.lag2 ', 'Probability contacting nucleosome core.lag1 ',
           'Probability contacting nucleosome core.lag2 ', 'Mobility to bend towards major groove.lag1 ',
           'Mobility to bend towards major groove.lag2 ', 'Slide3.lag1 ', 'Slide3.lag2 ', 'Slide2.lag1 ',
           'Slide2.lag2 ', 'Slide1.lag1 ', 'Slide1.lag2 ', 'Shift1.lag1 ', 'Shift1.lag2 ', 'Bend.lag1 ', 'Bend.lag2 ',
           'Rise1.lag1 ', 'Rise1.lag2 ', 'Rise stiffness.lag1 ', 'Rise stiffness.lag2 ',
           'Mobility to bend towards minor groove.lag1 ', 'Mobility to bend towards minor groove.lag2']

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_test[:2499])

shap.summary_plot(shap_values, x_test[:2499], feature_names=columns, max_display=30)
plt.savefig("shap.svg", dpi=500, format="svg")
