import numpy as np
import scipy.stats as stats


def comparebypwm(A, B):
    result_all = []
    count = 0
    for i in range(401):
        temp_A = A[:, i]
        temp_B = B[:, i]
        if i == 561:
            print('yes')
            result_all.append(1)
        else:
            res1, res2 = stats.mannwhitneyu(temp_A, temp_B, alternative='two-sided')

            print(i)
            print(res2)
            result_all.append(res2)
            if res2 < 0.0034:
                count += 1
    return result_all, count


cytoplasmMotif = np.loadtxt('/home/user_home/liuhaibin/code/mRNA_loc/data/motif/res/Cytoplasm_motif.txt')
nucleusMotif = np.loadtxt('/home/user_home/liuhaibin/code/mRNA_loc/data/motif/res/Nucleus_motif.txt')
result_all, count = comparebypwm(cytoplasmMotif, nucleusMotif)

fmotif = open("./data/motif/HOCOMOCOv11_core_pwms_HUMAN_mono.txt", 'r')
motifs = {}
for line in fmotif.readlines():
    if line[0] != ' ':
        if line[0] == '>':
            key = line.strip('>').strip('\n')
            a = []
        if line[0] != '>':
            a.append(list(line.upper().strip('\n').split("\t")))
            motifs[key] = a

for key in motifs.keys():
    motifs[key] = np.array(motifs[key], dtype="float64")

motifsName = list(motifs.keys())

result_dict = {k: v for k, v in zip(motifsName, result_all)}
result_dict = {k: v for k, v in sorted(result_dict.items(), key=lambda item: item[1])}

print(result_all)
