from torch.utils.data import Dataset
import numpy
import numpy as np
import pandas


def read_fasta(file):
    f = open(file)
    documents = f.readlines()
    string = ""
    flag = 0
    fea = []
    for document in documents:
        if document.startswith(">") and flag == 0:
            flag = 1
            continue
        elif document.startswith(">") and flag == 1:
            string = string.upper()
            fea.append(string)
            string = ""
        else:
            string += document
            string = string.strip()
            string = string.replace(" ", "")
    fea.append(string)
    f.close()
    return fea

class LDAMLncAtlasDataset(Dataset):
    def __init__(self, file_name, feature_path):
        test_mRna_label = []
        for file in file_name:
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
            test_mRna_label = test_mRna_label + label
        self.label = test_mRna_label

        self.feature = pandas.read_csv(
            feature_path,
            sep=',',
            header=None,
            index_col=None,
            low_memory=False).values.tolist()
        self.feature = numpy.array(self.feature)
        self.feature = self.feature[1:, 1:]
        print('Load successfully.')

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        item = {'label': self.label[idx], 'feature': self.feature[idx]}
        return item

    def get_labels(self):
        return self.label

    def get_cls_num_list(self):
        return np.bincount(self.get_labels())

    def get_feature_len(self):
        return len(self.feature[0])


