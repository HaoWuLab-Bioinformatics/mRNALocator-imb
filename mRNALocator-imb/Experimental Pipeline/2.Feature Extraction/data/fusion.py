import pandas as pd
import os
import shutil
from pathlib import Path

def feature_merge(feature_list, output):
    data_list = ['train', 'test', 'weight']
    
    for data in data_list:
        read_base_path = Path(f'../cache/{data}/')
        save_base_path = Path(f'./fusion/{data}/')
        
        save_base_path.mkdir(parents=True, exist_ok=True)
        save_file_path = save_base_path / output
        
        if len(feature_list) == 1:
            fea_name = feature_list[0]
            source_file = read_base_path / f'{fea_name}.csv'
            
            if not source_file.exists():
                raise FileNotFoundError(f"Source feature file not found: {source_file}")
            
            shutil.copy2(source_file, save_file_path)
            print(f"Single feature file copied: {source_file} -> {save_file_path}")
        
        else:
            df_list = []
            for fea in feature_list:
                fea_path = read_base_path / f'{fea}.csv'
                
                if not fea_path.exists():
                    raise FileNotFoundError(f"Feature file not found: {fea_path}")
                
                df = pd.read_csv(
                    fea_path,
                    sep=',',
                    header=0,
                    index_col=0
                )
                df_list.append(df)
            
            df_merged = pd.concat(df_list, axis=1)
            df_merged.to_csv(save_file_path, sep=",")
            print(f"Multiple features merged and saved: {save_file_path}")

def feature_combine():
    # RF-based Target Task Fused Features
    set1_feature_list = ['TPCP', 'Mismatch_k=5_m=1', 'RCKmer_k=4', 'PseKNC_k=3']
    feature_merge(set1_feature_list, output='set.csv')

    # TF-based Target Task Fused Features
    set2_feature_list = ['Kmer_k=8']
    feature_merge(set2_feature_list, output='set2.csv')

if __name__ == '__main__':
    feature_combine()