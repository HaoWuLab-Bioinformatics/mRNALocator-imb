import os
import glob
import random
import shutil
import matplotlib.pyplot as plt


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
RAW_DATA_DIR = "raw"  # Read fasta from this dir


def read_fasta(file_path):
    seqs, current = [], ""
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('>'):
                    if current: seqs.append(current)
                    current = line + '\n'
                else:
                    current += line + '\n'
            if current: seqs.append(current)
        return seqs, len(seqs)
    except Exception as e:
        print(f"Read error {file_path}: {e}")
        return [], 0


def extract_loc_name(file_name):
    name = file_name.replace('.fasta', '')
    if '_indep' in name:
        return name.split('_indep')[0]
    elif '_train' in name:
        return name.split('_train')[0]
    return None


def merge_fasta():
    fasta_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.fasta"))
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: {RAW_DATA_DIR} dir not found! Add fasta files in it first.")
        return [], ""
    if not fasta_files:
        print(f"Error: No .fasta files in {RAW_DATA_DIR}")
        return [], ""

    temp_dir = "temp_merged"  # Temp dir for merged files
    os.makedirs(temp_dir, exist_ok=True)
    loc_file_dict = {} 

    for file in fasta_files:
        loc = extract_loc_name(os.path.basename(file))
        if loc:
            loc_file_dict[loc] = loc_file_dict.get(loc, []) + [file]

    if not loc_file_dict:
        print("Error: No valid files (need _train/_indep suffix)")
        return [], temp_dir

    merged_files = []
    print("Merging files...")
    for loc, files in sorted(loc_file_dict.items()):
        merged_seqs, total = [], 0
        merged_fpath = os.path.join(temp_dir, f"{loc}.fasta")
        for file in files:
            seqs, cnt = read_fasta(file)
            merged_seqs.extend(seqs)
            total += cnt
        if merged_seqs:
            with open(merged_fpath, 'w') as f:
                f.writelines(merged_seqs)
            merged_files.append(merged_fpath)
            print(f"  {loc}.fasta: {total} seqs")

    return merged_files, temp_dir


def split_seqs(sequences, total):
    """Split seqs to train/weight/test (5:1:1)"""
    shuffled = random.sample(sequences, total)
    train_end = int(total * 5 / 7)
    weight_end = train_end + int(total * 1 / 7)
    return shuffled[:train_end], shuffled[train_end:weight_end], shuffled[weight_end:]


def save_fasta(sequences, save_path):
    """Save seqs to fasta file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        with open(save_path, 'w') as f:
            f.writelines(sequences)
        return True
    except Exception as e:
        print(f"Save error {save_path}: {e}")
        return False


def plot_split_result(summary):
    """Plot split distribution (300DPI)"""
    split_types = ['Train', 'Weight', 'Test']
    loc_names = [item[0] for item in summary]
    data = {'Train': [item[2] for item in summary],
            'Weight': [item[3] for item in summary],
            'Test': [item[4] for item in summary]}
    totals = [sum(data[st]) for st in split_types]

    # Plot settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c', '#9467bd']
    default_labels = ['Cytoplasm', 'ER', 'Extracellular', 'Mitochondria', 'Nucleus']

    bottom = [0, 0, 0]
    for i, loc in enumerate(loc_names):
        ax.bar(split_types, [data[st][i] for st in split_types], bottom=bottom,
               color=colors[i % len(colors)], label=default_labels[i%len(default_labels)] if i<5 else loc)
        bottom = [bottom[j] + data[split_types[j]][i] for j in range(3)]

    for i, total in enumerate(totals):
        ax.text(i, total + 150, f'Total: {total}', ha='center', fontsize=9, fontweight='bold')

    ax.set_title('Dataset Split Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sequence Count', fontsize=12)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, max(totals) * 1.15)

    save_path = 'split_distribution.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path}")


def main():
    """Main: merge -> split -> save -> plot"""
    merged_files, temp_dir = merge_fasta()
    if not merged_files or temp_dir == "":
        return

    print(f"\nSplitting (seed {RANDOM_SEED}, 5:1:1)...")
    summary = []
    for file in merged_files:
        loc = os.path.splitext(os.path.basename(file))[0]
        seqs, total = read_fasta(file)
        if total == 0:
            print(f"Warning: No seqs in {file}, skip")
            continue
        train, weight, test = split_seqs(seqs, total)
        save_fasta(train, f"train/{loc}.fasta")
        save_fasta(weight, f"weight/{loc}.fasta")
        save_fasta(test, f"test/{loc}.fasta")
        summary.append((loc, total, len(train), len(weight), len(test)))

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Clean up temp dir: {temp_dir}")

    if summary:
        print("\nSplit Summary (5:1:1)")
        print("-" * 60)
        print(f"{'Location':<20} {'Total':<6} {'Train':<6} {'Weight':<6} {'Test':<6}")
        print("-" * 60)
        t_all, tr_all, w_all, te_all = 0, 0, 0, 0
        for name, t, tr, w, te in summary:
            print(f"{name:<20} {t:<6} {tr:<6} {w:<6} {te:<6}")
            t_all += t; tr_all += tr; w_all += w; te_all += te
        print("-" * 60)
        print(f"{'Total':<20} {t_all:<6} {tr_all:<6} {w_all:<6} {te_all:<6}")
        print(f"Files saved to: train/, weight/, test/")
        plot_split_result(summary)


if __name__ == "__main__":
    main()