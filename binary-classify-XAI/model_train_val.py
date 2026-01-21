import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import copy

# --- Global config ---
DATA_PATH = "./all_species_data.csv" # You have change your data path
NUM_CLASSES = 2
SEQ_LEN = 70
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_FOLDER = "IG_CV_result"
MODEL_SAVE_PATH = './saved_models/best_cv_models.pth'

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
if not os.path.exists('./saved_models'):
    os.makedirs('./saved_models')

AA_LIST = list('AGILMPVFWYCNSTQDEHKR')
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
CLASS_NAMES = [f'C{i}' for i in range(0, NUM_CLASSES-1)] + ['Neg']

# --- Data process ---
def sampling_neg(all_df, neg_ratio=1):
    """
    For label_1 here:
        Plastid = 0; Cytosol = 1; ER = 2; Nucleus = 3; Peroxisome = 4
    """
    pos_df = all_df[all_df["label_1"] == 0].copy()
    pos_df["location"] = "plastid"
    pos_df["label_1"] = 0

    pos_count = len(pos_df)
    total_neg_to_sample = int(pos_count * neg_ratio)

    neg_df = all_df[all_df["label_1"] != 0].copy()
    sampling_classes = [1, 2, 3, 4] 
    neg_to_process = neg_df[neg_df["label_1"].isin(sampling_classes)]

    ideal_per_class = total_neg_to_sample / len(sampling_classes)

    def balanced_sample(group):
        n_actual = len(group)
        n_sample = min(n_actual, ideal_per_class)
        n_sample = int(np.floor(n_sample))
        return group.sample(n=n_sample, random_state=42)

    sampled_neg_initial = neg_to_process.groupby("label_1", group_keys=False).apply(balanced_sample)
    
    current_sampled_count = len(sampled_neg_initial)
    remaining_to_sample = total_neg_to_sample - current_sampled_count
    
    if remaining_to_sample > 0:
        sampled_indices = sampled_neg_initial.index
        unsampled_neg = neg_to_process[~neg_to_process.index.isin(sampled_indices)]
        if len(unsampled_neg) > 0:
            n_extra = min(remaining_to_sample, len(unsampled_neg))
            extra_samples = unsampled_neg.sample(n=n_extra, random_state=42)
            sampled_neg = pd.concat([sampled_neg_initial, extra_samples]).reset_index(drop=True)
        else:
            sampled_neg = sampled_neg_initial.reset_index(drop=True)
    else:
        sampled_neg = sampled_neg_initial.reset_index(drop=True)

    merged = pd.concat([pos_df, sampled_neg], axis=0).reset_index(drop=True)
    merged = merged[["species", "locus", "location", "seq", "label_1", "label_2"]]

    return merged

def make_cv_by_gene(df, n_splits=5, split_layer="label_2", random_state=42):
    """
    For label_2 here:
        Pos. seq = 1 (Plastid)
        Neg. seq = 0 (Cytosol, ER, Nucleus, Peroxisome)
    """

    df = df.copy()
    df["gene_id"] = df["locus"].str.replace(r"\.\d+$", "", regex=True)

    gene_df = df.groupby("gene_id").first()[[split_layer]].reset_index()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    gene_df["fold"] = -1
    for fold, (tr_idx, va_idx) in enumerate(skf.split(gene_df["gene_id"], gene_df[split_layer])):
        val_genes = gene_df.iloc[va_idx]["gene_id"]
        gene_df.loc[gene_df["gene_id"].isin(val_genes), "fold"] = fold

    gene2fold = dict(zip(gene_df["gene_id"], gene_df["fold"]))
    df["fold"] = df["gene_id"].map(gene2fold)

    return df

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

# --- Model architecture ---
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

# --- Train and Validation ---
def train_cv_and_save_models(df):
    total_samples = len(df)
    best_models_dict = {} 

    final_probs = np.zeros(total_samples, dtype=np.float32)
    
    fold_performance = []
    all_cv_preds = []
    all_cv_labels = []

    folds = sorted(df['fold'].unique())
    
    for fold in folds:
        print(f"\n{'='*10} Processing Fold {fold} {'='*10}")
        
        train_df = df[df['fold'] != fold]
        val_df = df[df['fold'] == fold]
        
        train_loader = DataLoader(ProteinDataset(train_df, label_layer="label_2"), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(ProteinDataset(val_df, label_layer="label_2"), batch_size=BATCH_SIZE, shuffle=False)
        
        model = CNNModel(num_classes=NUM_CLASSES).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) 
        
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())
        
        for epoch in range(EPOCHS):
            # Training
            model.train()
            train_loss = 0.0
            for batch_x, batch_y, _ in train_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            
            avg_train_loss = train_loss / len(train_loader.dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y, _ in val_loader:
                    batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
            
            avg_val_loss = val_loss / len(val_loader.dataset)
            scheduler.step()
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if (epoch+1) % 5 == 0:
                print(f" Fold {fold} | Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
        print(f" Fold {fold}: Loaded best model (Val Loss: {best_val_loss:.4f})")
        model.load_state_dict(best_model_wts)
        best_models_dict[int(fold)] = {k: v.cpu() for k, v in best_model_wts.items()}

        # Calculate Metrics
        model.eval()
        fold_preds, fold_labels = [], []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_idx in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                
                preds = torch.argmax(outputs, dim=1)
                fold_preds.extend(preds.cpu().tolist())
                fold_labels.extend(batch_y.cpu().tolist())
                
                probs = torch.nn.functional.softmax(outputs, dim=1)
                target_probs = probs.gather(1, batch_y.view(-1, 1)).squeeze()
                final_probs[batch_idx.cpu().numpy()] = target_probs.cpu().numpy()
        
        acc = accuracy_score(fold_labels, fold_preds)
        p, r, f1, _ = precision_recall_fscore_support(fold_labels, fold_preds, average='weighted', zero_division=0)

        fold_performance.append({
            'Fold': fold, 'Loss': best_val_loss, 'Accuracy': acc, 
            'Precision': p, 'Recall': r, 'F1-Score': f1
        })
        all_cv_preds.extend(fold_preds)
        all_cv_labels.extend(fold_labels)

    performance_df = pd.DataFrame(fold_performance)
    avg_performance = performance_df.drop(columns=['Fold']).mean()
    print("\n" + "="*50)
    print("Training Complete. Average Performance")
    print(avg_performance.to_string(float_format='%.4f'))
    print("="*50)
    
    cm = confusion_matrix(all_cv_labels, all_cv_preds)

    df['cv_prob'] = final_probs
    
    return df, performance_df, cm, best_models_dict

if __name__ == "__main__":    
    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}...")
        all_df = pd.read_csv(DATA_PATH)
        
        print("Preprocessing data...")
        model_data_df = sampling_neg(all_df, neg_ratio=1)
        print(f"Negative samples (label_2=0): {sum(model_data_df['label_2'] == 0)}")
        
        df_cross_val = make_cv_by_gene(model_data_df, n_splits=5, split_layer="label_2")
        
        print("Starting training...")
        df_result, performance_df, cm, saved_models = train_cv_and_save_models(df_cross_val)
        
        # Save model
        torch.save(saved_models, MODEL_SAVE_PATH)
        print(f"Models saved to {MODEL_SAVE_PATH}")

        result_csv_path = "df_result.csv"
        df_result.to_csv(result_csv_path, index=False)
        print(f"Result DataFrame saved to {result_csv_path}")
        
    else:
        print(f"Error: Data file not found at {DATA_PATH}")