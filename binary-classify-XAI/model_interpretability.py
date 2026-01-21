import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from captum.attr import IntegratedGradients
import os
import re
import matplotlib.pyplot as plt
import logomaker

# --- Global config (must be same as model_train_val.py) ---
BATCH_SIZE = 64
SEQ_LEN = 70
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IG_FOLDER = "./ig_reports"
MODEL_SAVE_PATH = './saved_models/best_cv_models.pth'
DATA_RESULT_PATH = "df_result.csv"

AA_LIST = list('AGILMPVFWYCNSTQDEHKR')
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

def seq_to_onehot(seq, max_len=SEQ_LEN):
    onehot = np.zeros((20, max_len), dtype=np.float32)
    for i, char in enumerate(seq[:max_len]):
        if char in AA_TO_IDX:
            onehot[AA_TO_IDX[char], i] = 1.0
    return onehot

class ProteinDataset(Dataset):
    def __init__(self, df, label_layer="label_2"):
        self.seqs = df['seq'].values
        self.labels = df[label_layer].values
        self.indices = df.index.values 

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_oh = seq_to_onehot(self.seqs[idx])
        label = self.labels[idx]
        return torch.tensor(seq_oh), torch.tensor(label, dtype=torch.long), idx

class CNNModel(nn.Module):
    def __init__(self, num_classes=2, input_channels=20, seq_len=70):
        super(CNNModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self._to_linear = self._calculate_flatten_size(input_channels, seq_len)
        
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _calculate_flatten_size(self, input_channels, seq_len):
        with torch.no_grad():
            x = torch.zeros(1, input_channels, seq_len)
            x = self.features(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --- IG analysis ---

def compute_class_specific_statistics(df, best_models_dict, num_classes=NUM_CLASSES):
    print("\n Starting Class-Specific IG Statistics Calculation & Prediction Collection...")
    
    all_preds = np.full(len(df), -1, dtype=int)
    
    global_sum = np.zeros((num_classes, 20, SEQ_LEN), dtype=np.float64)
    global_sq_sum = np.zeros((num_classes, 20, SEQ_LEN), dtype=np.float64)
    global_count = np.zeros((num_classes, 20, SEQ_LEN), dtype=np.float64)
    
    folds = sorted([int(k) for k in best_models_dict.keys()])
    
    model = CNNModel(num_classes=num_classes).to(DEVICE)
    ig = IntegratedGradients(model)
    
    for fold in folds:
        print(f" -> Processing Fold {fold}...")
        
        val_df = df[df['fold'] == fold]

        val_loader = DataLoader(ProteinDataset(val_df), batch_size=BATCH_SIZE, shuffle=False)
        
        # Load Fold best weight of model
        model.load_state_dict(best_models_dict[fold] if fold in best_models_dict else best_models_dict[str(fold)])
        model.eval()
        
        current_idx_pointer = 0
        
        for batch_x, batch_y, _ in tqdm(val_loader, leave=False):
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            batch_len = batch_x.size(0)

            real_indices = val_df.index[current_idx_pointer : current_idx_pointer + batch_len]
            current_idx_pointer += batch_len
            
            # 1. Get prediction
            with torch.no_grad():
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1)
            
            all_preds[real_indices] = preds.cpu().numpy()
            
            # 2. For each Class computes IG
            for target_class in range(num_classes):
                # Condition：Label == Class AND Pred == Class
                mask = (batch_y == target_class) & (preds == target_class)
                
                if not mask.any():
                    continue
                
                target_inputs = batch_x[mask]
                target_inputs.requires_grad = True
                
                # IG
                attr = ig.attribute(target_inputs, target=target_class, internal_batch_size=BATCH_SIZE)
                attr_np = attr.detach().cpu().numpy() 
                input_mask = target_inputs.detach().cpu().numpy()
                masked_attr = attr_np * input_mask
                
                global_sum[target_class] += masked_attr.sum(axis=0)
                global_sq_sum[target_class] += (masked_attr ** 2).sum(axis=0)
                global_count[target_class] += input_mask.sum(axis=0)

    if (all_preds == -1).any():
        print(f"⚠️ Warning: {(all_preds == -1).sum()} samples were not predicted (Check indices).")
    
    df['pred'] = all_preds
    print("Predictions stored in df['pred'] correctly.")

    # Calculate final Mean 和 Std
    print("Computing final IG statistics...")
    nonzero_mask = global_count > 0
    
    final_mean = np.zeros_like(global_sum)
    final_std = np.zeros_like(global_sum)
    
    final_mean[nonzero_mask] = global_sum[nonzero_mask] / global_count[nonzero_mask]
    
    variance = np.zeros_like(global_sum)
    variance[nonzero_mask] = (global_sq_sum[nonzero_mask] / global_count[nonzero_mask]) - (final_mean[nonzero_mask] ** 2)
    final_std[nonzero_mask] = np.sqrt(np.maximum(variance[nonzero_mask], 0))
    
    print("Statistics Calculation Complete.")
    
    return df, final_mean, final_std, global_count

def generate_cluster_reports_from_stats(class_means, class_stds, class_counts, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    num_classes = class_means.shape[0]
    seq_len = class_means.shape[2]

    for cluster_id in range(num_classes):
        print(f"Generating report for Cluster {cluster_id}...")
        
        aa_avg = class_means[cluster_id]
        aa_std = class_stds[cluster_id]
        aa_count = class_counts[cluster_id]
        
        if aa_count.sum() == 0:
            print(f"Warning: Cluster {cluster_id} has no valid samples.")
            continue

        df_avg = pd.DataFrame(aa_avg, index=AA_LIST, columns=range(1, seq_len+1))
        df_count = pd.DataFrame(aa_count, index=AA_LIST, columns=range(1, seq_len+1))
        df_std = pd.DataFrame(aa_std, index=AA_LIST, columns=range(1, seq_len+1))
        
        result_rows = []
        
        for pos in range(1, seq_len+1):
            col_avg = df_avg[pos]
            col_count = df_count[pos]
            col_std = df_std[pos]
            
            sorted_aas = col_avg.sort_values(ascending=False).index
            
            cell_texts = []
            for aa in sorted_aas:
                val = col_avg[aa]
                cnt = col_count[aa]
                std = col_std[aa]
                
                if cnt == 0: continue
                
                txt = f"{aa}:{val:.4f}\n({int(cnt)})\n({std:.4f})"
                cell_texts.append(txt)
            
            result_rows.append(cell_texts)
        
        max_depth = max([len(x) for x in result_rows]) if result_rows else 0
        final_grid = np.full((max_depth, seq_len), "", dtype=object)
        
        for i, cells in enumerate(result_rows): 
            for j, txt in enumerate(cells):     
                if j < max_depth:
                    final_grid[j, i] = txt
                
        final_df = pd.DataFrame(final_grid, columns=range(1, seq_len+1), index=range(1, max_depth+1))
        
        save_path = f"{save_folder}/Class_{cluster_id}_IG_Pattern.csv"
        final_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {save_path}")

    print("\n All reports generated.")

def compute_freq_matrix(seqs, aa_list, target_len=70):
    count = np.zeros((target_len, len(aa_list)))
    valid_seqs = [s for s in seqs if len(s) >= target_len]
    
    if len(valid_seqs) == 0:
        return pd.DataFrame(np.zeros((len(aa_list), target_len)), index=aa_list, columns=range(1, target_len+1))

    for seq in valid_seqs:
        for i in range(target_len):
            aa = seq[i]
            if aa in aa_list:
                j = aa_list.index(aa)
                count[i, j] += 1

    freq = count / len(valid_seqs)
    return pd.DataFrame(freq, columns=aa_list).T

def tidy_aa_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    aa_order = list("ACDEFGHIKLMNPQRSTVWY")
    df_raw.columns = [str(c) for c in df_raw.columns]
    data = {col: {aa: 0.0 for aa in aa_order} for col in df_raw.columns}
    pat = re.compile(r"^([A-Z]):([-\d\.eE]+)") 
    
    for col in df_raw.columns:
        for cell in df_raw[col]:
            cell_str = str(cell)
            match = pat.match(cell_str)
            if match:
                aa  = match.group(1)
                val = float(match.group(2))
                if aa in aa_order:
                    data[col][aa] = val

    tidy = pd.DataFrame(data).loc[aa_order]
    sorted_cols = sorted(tidy.columns, key=lambda x: int(x))
    tidy = tidy[sorted_cols]
    
    return tidy

def process_multiclass_weighted_ig(df_result, ig_folder_path, output_folder, num_classes, label_col):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if 'pred' not in df_result.columns:
        print("Error: 'pred' column not found in DataFrame. Please add prediction results.")
        return

    AA_FREQ_LIST = list("ACDEFGHIKLMNPQRSTVWY")

    for cluster_id in range(num_classes):
        print(f"\nProcessing Cluster {cluster_id} Weighted Analysis...")

        pos_mask = (df_result[label_col] == cluster_id) & (df_result['pred'] == cluster_id)
        neg_mask = (df_result[label_col] != cluster_id) & (df_result['pred'] == df_result[label_col])
        
        pos_seqs = df_result.loc[pos_mask, 'seq'].tolist()
        neg_seqs = df_result.loc[neg_mask, 'seq'].tolist()
        
        print(f"  - Positive Samples (Correct Class {cluster_id}): {len(pos_seqs)}")
        print(f"  - Negative Samples (Correct Other Classes): {len(neg_seqs)}")
        
        if len(pos_seqs) == 0:
            print(f"  - Skipping Cluster {cluster_id} (No positive samples)")
            continue

        pos_freq_df = compute_freq_matrix(pos_seqs, AA_FREQ_LIST, SEQ_LEN)
        neg_freq_df = compute_freq_matrix(neg_seqs, AA_FREQ_LIST, SEQ_LEN)
        
        freq_diff = pos_freq_df - neg_freq_df
        abs_freq_diff = freq_diff.abs()

        ig_file = os.path.join(ig_folder_path, f"Class_{cluster_id}_IG_Pattern.csv")
        if not os.path.exists(ig_file):
            print(f"  - IG File not found: {ig_file}")
            continue
            
        raw_ig_df = pd.read_csv(ig_file)
        cleaned_ig_df = tidy_aa_df(raw_ig_df) 
        
        cleaned_ig_df.columns = range(1, SEQ_LEN+1)
        abs_freq_diff.columns = range(1, SEQ_LEN+1)
        
        weighted_ig_df = cleaned_ig_df * abs_freq_diff

        save_path_matrix = os.path.join(output_folder, f"Class_{cluster_id}_Weighted_IG_Matrix.csv")
        weighted_ig_df.to_csv(save_path_matrix)
        
        print(f"  - Saved Weighted Matrix: {save_path_matrix}")

    print("\n All Weighted Analyses Completed.")

def plot_logo(weighted_ig_df, title="Sequence Logo", top_n=5, figsize=(10, 5), ylim=None, save_path="Logo_plot.png"):

    df = weighted_ig_df.copy()
    if df.shape[0] == 20:
        df = df.T
    df.index = df.index.astype(int)

    def filter_top_n(row):
        if row.abs().max() == 0:
            return row
        row[row.abs() < row.abs().nlargest(top_n).min()] = 0
        return row

    df = df.apply(filter_top_n, axis=1)
    aa_color = {
        'A': 'green','V': 'green','L': 'green','I': 'green','P': 'green','G': 'green','M': 'green',
        'F': 'purple','W': 'purple','Y': 'purple',
        'S': 'orange','T': 'orange','C': 'orange','N': 'orange','Q': 'orange',
        'D': 'red','E': 'red',
        'K': 'blue','R': 'blue','H': 'blue'
    }

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    logo = logomaker.Logo(
        df,
        color_scheme=aa_color,
        width=0.9,
        ax=ax
    )
    if ylim is None:
        pos_sum = df[df > 0].sum(axis=1).max()
        neg_sum = df[df < 0].sum(axis=1).min()
        limit = max(abs(pos_sum), abs(neg_sum)) * 1.1
        if limit == 0:
            limit = 1.0
        ax.set_ylim(-limit, limit)
    else:
        ymin, ymax = ylim
        ax.set_ylim(ymin, ymax)

    logo.style_spines(visible=True)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Importance", fontsize=10)

    plt.tight_layout()
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Logo plot save to: {save_path}")


if __name__ == "__main__":
    if not os.path.exists(DATA_RESULT_PATH) or not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Missing required files.")
        print(f"Please run 'train_val.py' first to generate '{DATA_RESULT_PATH}' and '{MODEL_SAVE_PATH}'")
    else:
        print("Loading data and models...")
        df_result = pd.read_csv(DATA_RESULT_PATH)
        saved_models = torch.load(MODEL_SAVE_PATH)

        # XAI calculation
        df_result['fold'] = df_result['fold'].astype(int)
        df_result, ig_means, ig_stds, ig_counts = compute_class_specific_statistics(df_result, saved_models)
        generate_cluster_reports_from_stats(ig_means, ig_stds, ig_counts, IG_FOLDER)
        process_multiclass_weighted_ig(df_result, IG_FOLDER, IG_FOLDER, NUM_CLASSES, 
                                       label_col="label_2") #Pos. seq = 1 (Plastid); Neg. seq = 0 (Others)

        # Plot
        df_weighted = pd.read_csv("./ig_reports/Class_1_Weighted_IG_Matrix.csv", index_col=0)
        plot_logo(df_weighted, title="Transit peptides Importance Logo Plot", top_n=10)