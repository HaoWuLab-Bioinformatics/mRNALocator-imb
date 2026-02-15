import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

metric_colors = {
    'ACC': '#1f77b4',
    'MCC': '#2ca02c',
    'Precision': '#ff7f0e',
    'Recall': '#d62728',
    'Fscore': '#9467bd'
}

def read_single_csv(file_path, target_columns):
    try:
        df_header = pd.read_csv(file_path, nrows=0)
        file_columns = df_header.columns.tolist()
        missing_cols = [col for col in target_columns if col not in file_columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        df = pd.read_csv(file_path, usecols=target_columns)
        return df, None
    except Exception as e:
        return None, str(e)

def find_best_value(df, method_prefix, metric='ACC'):
    method_df = df[df['Method'].str.contains(method_prefix, na=False, regex=False)]
    if method_df.empty:
        return None, None, None
    best_idx = method_df[metric].idxmax()
    best_row = method_df.loc[best_idx]
    best_name = best_row['Method']
    best_score = best_row[metric]
    return best_row, best_name, best_score

def split_method_and_params(method_full_name):
    if '_' in method_full_name:
        split_idx = method_full_name.index('_')
        method_name = method_full_name[:split_idx]
        params = method_full_name[split_idx+1:].replace('_', ';')
    else:
        method_name = method_full_name
        params = "-"
    return method_name, params

def plot_all_methods_multi_metrics(best_results):
    best_results_sorted = sorted(best_results, key=lambda x: x['full_metrics']['Fscore'], reverse=True)
    methods = [item['best_name'] for item in best_results_sorted]
    metrics = ['ACC', 'MCC', 'Precision', 'Recall', 'Fscore']
    metric_data = {m: [item['full_metrics'][m] for item in best_results_sorted] for m in metrics}
    
    fig_width = max(20, len(methods) * 1.2)
    x = np.arange(len(methods))
    width = 0.12
    fig, ax = plt.subplots(figsize=(fig_width, 14))
    
    for i, metric in enumerate(metrics):
        offset = (i - 2) * width
        ax.bar(x + offset, metric_data[metric], width, label=metric, 
               color=metric_colors[metric], alpha=0.8)
    
    ax.set_title('Multi-Metrics Comparison of All Methods (Sorted by Fscore)', fontsize=18)
    ax.set_xlabel('Method', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=60, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=12)
    
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.05)) 
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    plt.tight_layout()
    plt.savefig('./Comparison_Results.png', bbox_inches='tight', dpi=300)
    plt.show()
    return fig

if __name__ == "__main__":
    current_dir = os.path.join(os.getcwd(), '../record/train')
    target_columns = ['Method', 'ACC', 'MCC', 'Precision', 'Recall', 'Fscore']
    csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]
    best_results = []

    for csv_file in csv_files:
        file_path = os.path.join(current_dir, csv_file)
        df, error = read_single_csv(file_path, target_columns)
        if error:
            print(f"Error: {csv_file} - {error}")
            continue
        method_prefix = re.sub('_record.csv$', '', csv_file)
        best_row, best_name, best_acc = find_best_value(df, method_prefix)
        if best_row is not None:
            best_results.append({'best_name': best_name, 'full_metrics': best_row})

    if best_results:
        print("\nAll Methods Ranking (Sorted by Fscore)")
        print("-" * 126)
        header = (
            f"{'Rank':<4} "
            f"{'Method Name':<20} "
            f"{'Parameters':<20} "
            f"{'ACC':<10} "
            f"{'MCC':<10} "
            f"{'Precision':<12} "
            f"{'Recall':<10} "
            f"{'Fscore':<10}"
        )
        print(header)
        print("-" * 126)
        
        sorted_results = sorted(best_results, key=lambda x: x['full_metrics']['Fscore'], reverse=True)
        for idx, result in enumerate(sorted_results):
            m = result['full_metrics']
            method_name, params = split_method_and_params(result['best_name'])
            row = (
                f"{idx+1:<4} "
                f"{method_name:<20} "
                f"{params:<20} "
                f"{m['ACC']:<10.4f} "
                f"{m['MCC']:<10.4f} "
                f"{m['Precision']:<12.4f} "
                f"{m['Recall']:<10.4f} "
                f"{m['Fscore']:<10.4f}"
            )
            print(row)
        
        print(f"\nGenerating comparison chart (all {len(best_results)} methods)...")
        plot_all_methods_multi_metrics(best_results)
        
        global_best = max(best_results, key=lambda x: x['full_metrics']['Fscore'])
        gb_m = global_best['full_metrics']
        gb_name, gb_params = split_method_and_params(global_best['best_name'])
        print(f"\nGlobal Best: {gb_name} ({gb_params})")
        print(f"ACC: {gb_m['ACC']:.4f} | MCC: {gb_m['MCC']:.4f} | Precision: {gb_m['Precision']:.4f} | Recall: {gb_m['Recall']:.4f} | Fscore: {gb_m['Fscore']:.4f}")
    else:
        print("No valid data found")