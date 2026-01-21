import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sparrow import read_fasta, Protein
from sparrow import predictors
from sparrow.predictors import batch_predict
from scipy.stats import mannwhitneyu
import os

# ==========================================
# Configuration
# ==========================================
INPUT_CSV = "./consensus_labels.csv"
ALL_SPECIES_CSV = "./all_species_data.csv" # Your seq data
MAX_LEN = 100
WINDOW_SIZE = 8
OUTPUT_DIR = "./seq_feature_fig/"  # Directory to save plots

# ==========================================
# Helper Functions
# ==========================================

def pval_to_star(pval):
    """Convert p-value to significance stars."""
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    else:
        return "ns"

def linear_sequence_profile_dataset_mean(seq_dict, feature_name: str, window_size: int = 8, max_len=100) -> pd.Series:
    """
    Returns a pd.Series, index=1..max_len, values are the mean of the feature 
    at each position in the dataset (nan indicates sequence is too short at that position, 
    np.nanmean excludes them).
    """
    profiles = []
    for P in seq_dict.values():
        prof = np.asarray(
            P.linear_sequence_profile(feature_name, window_size=window_size),
            dtype=float
        )
        pad = np.full(max_len, np.nan, dtype=float)
        # Handle cases where profile is longer than max_len or shorter
        length = min(len(prof), max_len)
        pad[:length] = prof[:length]
        profiles.append(pad)

    arr = np.vstack(profiles)             # shape = (num_seq, max_len)
    mean_by_pos = np.nanmean(arr, axis=0) # length max_len
    std_by_pos = np.nanstd(arr, axis=0)
    
    return pd.DataFrame({
        "position": np.arange(1, max_len + 1),
        "mean": mean_by_pos,
        "std": std_by_pos
    }).set_index("position")

def linear_sequence_profile_dataset_stats(seq_dict, feature_name: str, window_size: int = 8, max_len=100):
    """
    Returns (global_means, pos_mean_df)
    global_means: list, global average for each sequence
    pos_mean_df: DataFrame, mean/std for each position
    """
    profiles = []
    globals_ = []

    for P in seq_dict.values():
        prof = np.asarray(
            P.linear_sequence_profile(feature_name, window_size=window_size),
            dtype=float
        )
        # Global mean for each sequence (truncated to max_len)
        globals_.append(np.nanmean(prof[:max_len]))  

        pad = np.full(max_len, np.nan, dtype=float)
        length = min(len(prof), max_len)
        pad[:length] = prof[:length]
        profiles.append(pad)

    arr = np.vstack(profiles)
    mean_by_pos = np.nanmean(arr, axis=0)
    std_by_pos = np.nanstd(arr, axis=0)

    pos_df = pd.DataFrame({
        "position": np.arange(1, max_len + 1),
        "mean": mean_by_pos,
        "std": std_by_pos
    }).set_index("position")

    return globals_, pos_df

def run_linear_profiles_for_groups(group_dicts, feature_name, window_size=8, max_len=100):
    all_records = []
    pos_profiles = {}

    for group, seq_dict in group_dicts.items():
        globals_, pos_df = linear_sequence_profile_dataset_stats(seq_dict, feature_name, window_size, max_len)
        
        # Save global mean (for boxplot)
        for val in globals_:
            all_records.append({
                "Group": group,
                "MeanValue": val
            })
        
        # Save positional mean profile (for line plot)
        pos_profiles[group] = pos_df

    global_table = pd.DataFrame(all_records)
    return global_table, pos_profiles

def plot_multi_profiles(df_list, feature, label_list=None, save_path=None):
    n = len(df_list)
    # Use Set2, but if n > 8, cycle or extend palette
    palette = sns.color_palette("Set2", n if n <= 8 else n)

    plt.figure(figsize=(10, 5))

    for i, df in enumerate(df_list):
        label = label_list[i] if label_list else None
        color = palette[i]
        x = df.index
        y = df["mean"]
        # sd = df["std"] 

        plt.plot(x, y, label=label, color=color, linewidth=2)
        # plt.fill_between(x, y - sd, y + sd, alpha=0.15, color=color)

    plt.title(f"{feature} profile across positions")
    plt.xlabel("Position")
    plt.ylabel(f"{feature} score")
    plt.grid(alpha=0.3)
    if label_list:
        plt.legend()
    plt.tight_layout()
    
    if save_path:
        print(f"Saving profile plot to {save_path}...")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def box_plot_with_one_vs_rest(global_table, predictor_name, order=None, 
                              palette_map=None, y_pad=0, save_path=None, 
                              cluster_col="Group", val_col="MeanValue"):
    df = global_table.copy()
    
    # Ensure order is set
    if order is None:
        order = sorted(df[cluster_col].unique())
    
    # Ensure palette is set
    if palette_map is None:
        colors = sns.color_palette("Set2", n_colors=len(order))
        palette_map = dict(zip(order, colors))

    df[cluster_col] = pd.Categorical(df[cluster_col], categories=order, ordered=True)

    plt.figure(figsize=(6, 5))
    ax = sns.boxplot(
        data=df, x=cluster_col, y=val_col,
        order=order, hue=cluster_col, hue_order=order,
        palette=palette_map, width=0.6, dodge=False
    )
    if ax.legend_ is not None:
        ax.legend_.remove()

    results = []
    for target in order:
        vals_target = df[df[cluster_col] == target][val_col].dropna().values
        vals_rest   = df[df[cluster_col] != target][val_col].dropna().values
        if len(vals_target) > 0 and len(vals_rest) > 0:
            stat, pval = mannwhitneyu(vals_target, vals_rest, alternative="two-sided")
            results.append((target, pval))

    # Find the max value for each cluster, place asterisk above
    y_max_global = df[val_col].max()
    y_min_global = df[val_col].min()
    y_range = y_max_global - y_min_global if y_max_global != y_min_global else 1.0

    for i, (cluster, pval) in enumerate(results):
        star = pval_to_star(pval)
        cluster_data = df[df[cluster_col] == cluster][val_col]
        if cluster_data.empty:
            continue
        y_max_cluster = cluster_data.max()
        
        # Calculate text position
        text_y = y_max_cluster + (y_range * y_pad)
        
        ax.text(i, text_y,
                star, ha="center", va="bottom", fontsize=12, color="red")

    ax.set_title(f"Distribution of {predictor_name} across clusters")
    ax.set_ylabel(f"{predictor_name} score")
    ax.set_xlabel("Cluster")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        print(f"Saving boxplot to {save_path}...")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return pd.DataFrame(results, columns=["Cluster", "p-value"])

# ── Predictor definitions -------------------------------------------
PREDICTOR_FUNCS = {
    "coil":                 lambda P: P.predictor.dssp_coil(),                          # 0/1
    "α-helix":              lambda P: P.predictor.dssp_helicity(),                      # 0/1
    "β-strand":             lambda P: P.predictor.dssp_extended(),                      # 0/1

    "nes":                  lambda P: P.predictor.nuclear_export_signal(),              # 0-1 score
    "nis":                  lambda P: P.predictor.nuclear_import_signal(),              # 0-1 score

    "S_phosphorylation":    lambda P: P.predictor.serine_phosphorylation(),             # 0/1
    "T_phosphorylation":    lambda P: P.predictor.threonine_phosphorylation(),          # 0/1
    "Y_phosphorylation":    lambda P: P.predictor.tyrosine_phosphorylation(),           # 0/1

    "pscore":               lambda P: P.predictor.pscore(),                             # 0-1 score
    "tad":                  lambda P: P.predictor.transactivation_domains(),            # 0-1 score
    "mitochondrial_targeting":
                            lambda P: P.predictor.mitochondrial_targeting_sequence(),   # 0/1
    "transmembrane_region": lambda P: P.predictor.transmembrane_regions(),              # 0/1
    "IDR":                  lambda P: P.predictor.disorder(),                           # 0-1 score
}

def dataset_predictor_stats(seq_dict, pred_name, max_len=100):
    """
    Returns (global_mean, pos_mean[1..max_len])
    """
    get_vec = PREDICTOR_FUNCS[pred_name]
    mats, globals_ = [], []

    for P in seq_dict.values():
        vec = np.asarray(get_vec(P), dtype=float)
        # Global mean (average over the specific length)
        globals_.append(np.nanmean(vec[:max_len]))

        pad = np.full(max_len, np.nan) # Align positions (pad with nan if shorter)
        length = min(len(vec), max_len)
        pad[:length] = vec[:length]
        mats.append(pad)

    mats = np.vstack(mats)                        # shape = (n_seq, max_len)
    pos_mean = np.nanmean(mats, axis=0)           # length max_len
    return globals_, pos_mean

def run_predictors_for_groups(group_dicts, predictors_to_run=None, max_len=100):
    if predictors_to_run is None:
        predictors_to_run = list(PREDICTOR_FUNCS.keys())

    label_list = list(group_dicts.keys())

    all_records = []
    pos_profile = {g: {} for g in label_list}

    for group in label_list:
        seq_dict = group_dicts[group]
        for pred in predictors_to_run:
            globals_, p_mean = dataset_predictor_stats(seq_dict, pred, max_len=max_len)
            for value in globals_:
                all_records.append({
                    "Group": group,
                    "MeanValue": value,
                    "Predictor": pred
                })
            pos_profile[group] = pd.Series(p_mean,
                                           index=range(1, max_len+1),
                                           name=f"{group}")
    
    global_table = pd.DataFrame(all_records)
    return global_table, pos_profile

def plot_pos_profile(pos_profile, predictor_name, order=None, palette_map=None, save_path=None):
    plt.figure(figsize=(10, 5))
    
    if order is None:
        order = sorted(pos_profile.keys())
    
    if palette_map is None:
        palette_map = dict(zip(order, sns.color_palette("Set2", len(order))))

    for c in order:
        if c not in pos_profile:
            continue
        y = pos_profile[c]
        if isinstance(y, dict):
            y = pd.Series(y)
        # Check if y is empty or all NaN
        if y.empty or y.isna().all():
            continue
            
        plt.plot(y.index, y.values, label=f"{c}", color=palette_map.get(c, 'black'), linewidth=2)

    plt.title(f"{predictor_name} profile across positions")
    plt.xlabel("Position")
    plt.ylabel(f"{predictor_name} score")
    plt.grid(alpha=0.3)
    plt.legend(title="Cluster")
    plt.tight_layout()
    
    if save_path:
        print(f"Saving profile plot to {save_path}...")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def predict_ALBATROSS(df, mode="full", max_len=100):
    if mode == "full":
        seq_dict = {row['locus']: Protein(row['seq']) for _, row in df.iterrows()}
    elif mode == "trunc":
        seq_dict = {row['locus']: Protein(row['seq'][:max_len]) for _, row in df.iterrows()}

    networks = {
        "Rg_scaled": "scaled_rg",
        "Re_scaled": "scaled_re",
        "asphericity": "asphericity",
        "prefactor": "prefactor",
        "scaling_exponent": "scaling_exponent",
    }

    print(f"Running ALBATROSS predictions ({mode})...")
    for col, net in networks.items():
        result = batch_predict.batch_predict(
            seq_dict, network=net, batch_size=32
        )
        # result: {id: (something, value)}
        df[col] = df['locus'].map(lambda pid: result.get(pid, [None, None])[1])
    return df

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Check inputs
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        exit(1)
    if not os.path.exists(ALL_SPECIES_CSV):
        print(f"Error: {ALL_SPECIES_CSV} not found.")
        exit(1)

    print("Loading data...")
    df = pd.read_csv(INPUT_CSV)
    all_df = pd.read_csv(ALL_SPECIES_CSV)

    # Prepare negative control (non-plastid)
    neg_df = all_df[all_df["location"] != "plastid"].copy()
    
    # Sample negative control
    n_sample = min(10000, len(neg_df))
    neg_sub = neg_df[['species', 'locus', 'seq']].sample(n_sample, random_state=42).copy()
    neg_sub['cluster'] = -1
    
    # Merge
    merged_df = pd.concat([df, neg_sub], ignore_index=True)

    # Identify clusters
    # Get positive clusters sorted
    pos_clusters = sorted([c for c in merged_df['cluster'].unique() if c != -1])
    # Full list including -1 at the end
    all_cluster_ids = pos_clusters + [-1]
    
    # Create labels
    label_list = [f"Cluster {c}" for c in all_cluster_ids]
    
    # Prepare dictionary for processing
    print("Preparing protein sequences...")
    plastid_cluster_dict = {
        f"Cluster {cluster}": {
            row['locus']: Protein(row['seq'][:MAX_LEN])
            for _, row in merged_df[merged_df['cluster'] == cluster].iterrows()
        }
        for cluster in all_cluster_ids
    }

    cluster_counts = merged_df.groupby('cluster')['seq'].count()
    print("Cluster counts:\n", cluster_counts)

    # Define Palette
    PALETTE_CLU = dict(zip(label_list, sns.color_palette("Set2", n_colors=len(label_list))))

    # ---------------------------------------------------------
    # 1. Physicochemical Properties (Profiles & Boxplots)
    # ---------------------------------------------------------
    features = [
        "positive", "negative", "FCR", "NCPR", "kappa",
        "aromatic", "aliphatic", "polar", "proline", 
        "hydrophobicity", "seg-complexity"
    ]
    
    # Mapping for cleaner titles if needed, otherwise use key
    feature_titles = {
        "positive": "Fraction positive",
        "negative": "Fraction negative",
        "seg-complexity": "Seg-complexity"
    }

    # Y-padding adjustments for significance stars
    y_pads = {
        "kappa": 0, "aromatic": 0.005, "aliphatic": 0.001,
        "polar": 0.005, "proline": 0, "hydrophobicity": 0.001, 
        "seg-complexity": 0
    }

    print("\n--- Running Physicochemical Analysis ---")
    for feat in features:
        print(f"Processing {feat}...")
        display_name = feature_titles.get(feat, feat)
        
        # 1. Profile
        df_profiles = {
            name: linear_sequence_profile_dataset_mean(seq_dict, feature_name=feat, window_size=WINDOW_SIZE, max_len=MAX_LEN)
            for name, seq_dict in plastid_cluster_dict.items()
        }
        df_list = [df_profiles[label] for label in label_list]
        
        plot_multi_profiles(
            df_list, 
            feature=display_name, 
            label_list=label_list,
            save_path=os.path.join(OUTPUT_DIR, f"profile_{feat}.png")
        )

        # 2. Boxplot
        global_table, pos_profiles = run_linear_profiles_for_groups(
            plastid_cluster_dict, feature_name=feat, window_size=WINDOW_SIZE, max_len=MAX_LEN
        )
        
        pad = y_pads.get(feat, 0.05)
        stats_df = box_plot_with_one_vs_rest(
            global_table, 
            predictor_name=display_name, 
            order=label_list,
            palette_map=PALETTE_CLU,
            y_pad=pad,
            save_path=os.path.join(OUTPUT_DIR, f"boxplot_{feat}.png")
        )
        # print(stats_df) # Optional: print stats to console

    # ---------------------------------------------------------
    # 2. Structural/Signal Predictors (Profiles & Boxplots)
    # ---------------------------------------------------------
    predictors_list = [
        ("coil", "Coil"),
        ("α-helix", "α-helix"),
        ("β-strand", "β-strand"),
        ("nes", "NES"),
        ("nis", "NIS"),
        ("S_phosphorylation", "S phosphorylation"),
        ("T_phosphorylation", "T phosphorylation"),
        ("Y_phosphorylation", "Y phosphorylation"),
        ("pscore", "p-score"),
        ("tad", "Transactivation domains"),
        ("mitochondrial_targeting", "Mitochondrial targeting"),
        ("transmembrane_region", "Transmembrane region"),
        ("IDR", "IDR")
    ]

    print("\n--- Running Structural/Signal Predictors ---")
    for pred_key, pred_name in predictors_list:
        print(f"Processing {pred_name}...")
        
        # Calculate
        global_table, pos_profile = run_predictors_for_groups(
            plastid_cluster_dict, predictors_to_run=[pred_key], max_len=MAX_LEN
        )
        
        # Plot Profile
        plot_pos_profile(
            pos_profile, 
            predictor_name=pred_name,
            order=label_list,
            palette_map=PALETTE_CLU,
            save_path=os.path.join(OUTPUT_DIR, f"profile_pred_{pred_key}.png")
        )
        
        # Plot Boxplot
        stats_df = box_plot_with_one_vs_rest(
            global_table, 
            predictor_name=pred_name, 
            order=label_list,
            palette_map=PALETTE_CLU,
            y_pad=0.05,
            save_path=os.path.join(OUTPUT_DIR, f"boxplot_pred_{pred_key}.png")
        )

    # ---------------------------------------------------------
    # 3. ALBATROSS Biophysical Predictions
    # ---------------------------------------------------------
    print("\n--- Running ALBATROSS Analysis ---")
    # Need to update merged_df with 'consensus_cluster' column formatted like "Cluster X"
    # to match the logic of previous steps, or just use the existing 'cluster' int col
    # The box_plot_with_one_vs_rest function used previously expects 'Group' and 'MeanValue' columns.
    # The original notebook had a separate function for ALBATROSS. Let's use the generic one we made.
    
    ALBATROSS_df = predict_ALBATROSS(merged_df, mode="full", max_len=MAX_LEN)
    
    albatross_features = ["Rg_scaled", "Re_scaled", "asphericity", "prefactor", "scaling_exponent"]
    
    # Create a Group column for plotting
    ALBATROSS_df["Group"] = ALBATROSS_df["cluster"].apply(lambda x: f"Cluster {x}")
    
    for feat in albatross_features:
        print(f"Processing ALBATROSS {feat}...")
        
        # Rename columns to match what box_plot_with_one_vs_rest expects
        # We need a dataframe with 'Group' and 'MeanValue'
        temp_df = ALBATROSS_df[["Group", feat]].rename(columns={feat: "MeanValue"})
        
        box_plot_with_one_vs_rest(
            temp_df,
            predictor_name=feat,
            order=label_list,
            palette_map=PALETTE_CLU,
            y_pad=0.05,
            save_path=os.path.join(OUTPUT_DIR, f"boxplot_albatross_{feat}.png")
        )

    print("All processing done.")