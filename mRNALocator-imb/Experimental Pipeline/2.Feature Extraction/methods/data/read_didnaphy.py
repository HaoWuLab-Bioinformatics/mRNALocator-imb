import pickle

import numpy as np

file_name = '../data/didnaPhyche.data'


with open(file_name, 'rb') as handle:
    property_dict = pickle.load(handle)
property_name = property_dict.keys()

for p_name in property_name:
    tmp = np.array(property_dict[p_name], dtype=float)
    pmean = np.average(tmp)
    pstd = np.std(tmp)
    property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]


with open('my_file.csv', 'w') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in property_dict.items()]

print(property_name)