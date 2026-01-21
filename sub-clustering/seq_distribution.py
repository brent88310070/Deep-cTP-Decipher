import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import seaborn as sns
from collections import Counter
import os

# ==========================================
# Configuration & Constants
# ==========================================
INPUT_CSV = "./consensus_labels.csv"
SEQ_LENGTH = 100
WINDOW_SIZE = 8
OUTPUT = "./result_fig"

# Amino acid order and categories
AA_ORDER = [
    "A", "G", "I", "L", "M", "P", "V",  # Hydrophobic
    "F", "W", "Y",                      # Aromatic
    "C", "N", "S", "T", "Q",            # Polar
    "D", "E",                           # Negative charge
    "H", "K", "R"                       # Positive charge
]

AA_CATEGORIES = {
    "Hydrophobic": ["A", "G", "I", "L", "M", "P", "V"],
    "Aromatic": ["F", "W", "Y"],
    "Polar": ["C", "N", "S", "T", "Q"],
    "Negative charge": ["D", "E"],
    "Positive charge": ["H", "K", "R"]
}

CATEGORY_COLORS = {
    "Hydrophobic": "green",
    "Aromatic": "purple",
    "Polar": "orange",
    "Negative charge": "red",
    "Positive charge": "blue"
}

# Generate color mapping for Y-axis labels
AA_COLORS_LIST = []
for aa in AA_ORDER:
    found = False
    for category, aa_list in AA_CATEGORIES.items():
        if aa in aa_list:
            AA_COLORS_LIST.append(CATEGORY_COLORS[category])
            found = True
            break
    if not found:
        AA_COLORS_LIST.append("black") # Fallback

# ==========================================
# Helper Functions
# ==========================================

def get_cluster_seqs(df, cluster_id, N=100):
    """Extract sequences of length N for a specific cluster."""
    subset = df[df["cluster"] == cluster_id].copy()
    subset["seq_fragment"] = subset["seq"].str[:N]
    return subset[["locus", "cluster", "seq"]]

def plot_cluster_logo(sequences, title="Sequence Logo", aa_order="ACDEFGHIKLMNPQRSTVWY", 
                      aa_color_dict=None, pseudocount=0.0, max_len=None, 
                      dpi=300, figsize=(20, 6), save_path=None):
    """
    Generates and saves a sequence logo plot.
    """
    if aa_color_dict is None:
        aa_color_dict = {
            # Hydrophobic (green)
            'A':'green','V':'green','L':'green','I':'green','P':'green','G':'green','M':'green',
            # Aromatic (purple)
            'F':'purple','W':'purple','Y':'purple',
            # Polar (orange)
            'S':'orange','T':'orange','C':'orange','N':'orange','Q':'orange',
            # Negative (red)
            'D':'red','E':'red',
            # Positive (blue)
            'K':'blue','R':'blue','H':'blue'
        }

    sequences = [s for s in sequences if isinstance(s, str) and len(s) > 0]
    if len(sequences) == 0:
        print(f"[plot_cluster_logo] No sequences available for {title}.")
        return

    if max_len is None:
        L = max(len(s) for s in sequences)
    else:
        L = int(max_len)

    alphabet = list(aa_order)
    rows = []

    for i in range(L):
        counts = {aa: 0 for aa in alphabet}
        for seq in sequences:
            if i < len(seq):
                aa = seq[i]
                if aa in counts:
                    counts[aa] += 1

        # Smoothing + Normalization (Default: Off)
        total = sum(counts.values())
        if pseudocount > 0:
            total += pseudocount * len(alphabet)
            freqs = {aa: (counts[aa] + pseudocount) / total for aa in alphabet}
        else:
            if total == 0:
                freqs = {aa: 0.0 for aa in alphabet}
            else:
                freqs = {aa: counts[aa] / total for aa in alphabet}

        rows.append(freqs)

    count_matrix = pd.DataFrame(rows, columns=alphabet)

    plt.figure(figsize=figsize)
    logo = logomaker.Logo(count_matrix, color_scheme=aa_color_dict)
    logo.fig.set_dpi(dpi)
    logo.ax.set_title(title)
    logo.ax.set_xlabel("Position")
    logo.ax.set_ylabel("Frequency")
    plt.tight_layout()
    
    if save_path:
        print(f"Saving logo to {save_path}...")
        plt.savefig(save_path)
        plt.close(logo.fig) # Close specific figure
    else:
        plt.show()

def calculate_frequencies_window(sequences, seq_length=100, window_size=8):
    """Calculate amino acid frequencies using a sliding window."""
    n_windows = seq_length - window_size + 1
    window_frequencies = {w: Counter() for w in range(n_windows)}

    for seq in sequences:
        # Ensure sequence is long enough
        if len(seq) < seq_length:
            continue
            
        for w in range(n_windows):
            window_seq = seq[w : w + window_size]
            for aa in window_seq:
                window_frequencies[w][aa] += 1

    # Convert to DataFrame
    freq_df = pd.DataFrame(window_frequencies).fillna(0).T
    
    # Normalize (avoid division by zero)
    row_sums = freq_df.sum(axis=1)
    # If a window has 0 counts (no seqs), keep it 0
    freq_df = freq_df.div(row_sums.replace(0, 1), axis=0)
    
    freq_df = freq_df.T
    return freq_df.reindex(AA_ORDER, axis=0).fillna(0)

def plot_seq_distribution(df_clusters, pos=0, seq_len=100, mode="difference",
                          vmin=None, vmax=None, min_diff=0.01, save_path=None):
    """
    Plots the distribution difference between a specific cluster and all other clusters.
    """
    # Positive sequences (Target Cluster)
    positive_sequences = df_clusters[df_clusters['cluster'] == pos]['seq']
    positive_freq_df = calculate_frequencies_window(positive_sequences, seq_len, WINDOW_SIZE)

    # Negative sequences (Other clusters)
    negative_sequences = df_clusters[df_clusters['cluster'] != pos]['seq']
    negative_freq_df = calculate_frequencies_window(negative_sequences, seq_len, WINDOW_SIZE)
    
    pseudo = 1e-6
    if mode == "difference":
        result_df = positive_freq_df - negative_freq_df
        cmap = "coolwarm"
        center = 0
        cbar_label = "Frequency Difference"
        if vmin is None: vmin = -0.15
        if vmax is None: vmax = 0.15
    elif mode == "foldchange":
        diff = positive_freq_df - negative_freq_df
        log2fc = np.log2((positive_freq_df + pseudo) / (negative_freq_df + pseudo))
        log2fc[diff.abs() < min_diff] = 0
        result_df = log2fc
        cmap, center, cbar_label = f"Log2 Fold Change (|Δ| ≥ {min_diff})"
        vmin = vmin if vmin is not None else -2
        vmax = vmax if vmax is not None else 2
    elif mode == "weightedFC":
        diff = positive_freq_df - negative_freq_df
        log2fc = np.log2((positive_freq_df + pseudo) / (negative_freq_df + pseudo))
        result_df = diff * abs(log2fc)
        cmap, center, cbar_label = "Weighted FC (Δ × abs(log2FC))"
        vmin = vmin if vmin is not None else -0.2
        vmax = vmax if vmax is not None else 0.2
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Plot heatmap
    plt.figure(figsize=(10, 6), dpi=150)
    ax = sns.heatmap(result_df, cmap=cmap, center=center,
                     vmin=vmin, vmax=vmax,
                     cbar_kws={"label": cbar_label})

    # Customize y-axis labels with category colors
    for tick_label, color in zip(ax.get_yticklabels(), AA_COLORS_LIST):
        tick_label.set_color(color)
        tick_label.set_rotation(0)

    plt.xlabel("Position")
    plt.ylabel("Amino Acid")
    plt.title(f"{cbar_label} (Cluster {pos} vs others)")
    plt.xticks(range(0, seq_len+1 - WINDOW_SIZE, 10), labels=range(0, seq_len+1 - WINDOW_SIZE, 10), rotation=0)
    plt.tight_layout()
    
    if save_path:
        print(f"Saving heatmap to {save_path}...")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        exit(1)

    print(f"Reading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Identify all unique clusters
    unique_clusters = sorted(df['cluster'].unique())
    print(f"Found clusters: {unique_clusters}")

    # 1. Generate Sequence Logos for each cluster
    print("--- Generating Sequence Logos ---")
    for cluster in unique_clusters:
        seq_data = get_cluster_seqs(df, cluster_id=cluster, N=SEQ_LENGTH)['seq']
        filename = f"{OUTPUT}/logo_cluster_{cluster}.png"
        plot_cluster_logo(
            seq_data, 
            title=f"Sequence Logo (Cluster {cluster})", 
            max_len=SEQ_LENGTH,
            save_path=filename
        )

    # 2. Generate Distribution Heatmaps for each cluster
    print("--- Generating Distribution Heatmaps ---")
    for cluster in unique_clusters:
        filename = f"{OUTPUT}/heatmap_cluster_{cluster}_difference.png"
        plot_seq_distribution(
            df, 
            pos=cluster, 
            seq_len=SEQ_LENGTH, 
            mode="difference",
            save_path=filename
        )
    
    print("All processing done.")