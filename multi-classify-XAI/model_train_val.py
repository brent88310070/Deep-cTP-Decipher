import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy

# --- Global config ---
DATA_PATH = "./consensus_labels_core.csv" 
NUM_CLASSES = 5 # Based on your cluster num.
SEQ_LEN = 70
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_FOLDER = "ig_reports"
MODEL_SAVE_PATH = './saved_models/best_cv_models.pth'
RESULT_CSV_PATH = "df_result.csv"
CM_SAVE_PATH = "confusion_matrix.png"

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
if not os.path.exists('./saved_models'):
    os.makedirs('./saved_models')

# Amino acid mapping
AA_LIST = list('AGILMPVFWYCNSTQDEHKR')
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
CLASS_NAMES = [f'C{i}' for i in range(0, NUM_CLASSES)]

# --- Data process ---
def make_5fold_by_gene(df, n_splits=5, random_state=42):
    df = df.copy()
    df["gene_id"] = df["locus"].str.replace(r"\.\d+$", "", regex=True)

    gene_df = df.groupby("gene_id").first()[["cluster"]].reset_index()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    gene_df["fold"] = -1
    for fold, (tr_idx, va_idx) in enumerate(skf.split(gene_df["gene_id"], gene_df["cluster"])):
        val_genes = gene_df.iloc[va_idx]["gene_id"]
        gene_df.loc[gene_df["gene_id"].isin(val_genes), "fold"] = fold

    gene2fold = dict(zip(gene_df["gene_id"], gene_df["fold"]))
    df["fold"] = df["gene_id"].map(gene2fold)

    return df

def seq_to_onehot(seq, max_len=SEQ_LEN):
    # Initialize (20, L)
    onehot = np.zeros((20, max_len), dtype=np.float32)
    for i, char in enumerate(seq[:max_len]):
        if char in AA_TO_IDX:
            onehot[AA_TO_IDX[char], i] = 1.0
    return onehot

class ProteinDataset(Dataset):
    def __init__(self, df):
        self.seqs = df['seq'].values
        self.labels = df['cluster'].values
        self.indices = df.index.values # Track original indices

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_oh = seq_to_onehot(self.seqs[idx])
        label = self.labels[idx]
        return torch.tensor(seq_oh), torch.tensor(label, dtype=torch.long), idx

# --- Model architecture ---
class CNNModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, input_channels=20, seq_len=SEQ_LEN):
        super(CNNModel, self).__init__()
        
        # --- 1. Feature extraction layers (3 Layers Conv1D) ---
        # Layer 1: Large kernel (9) to capture broad features
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
            
            # Layer 2: Mid-level features (k=5)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
            
            # Layer 3: High-level abstract features (k=3)
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        # Automatically calculate flatten size
        self._to_linear = None
        self._calculate_flatten_size(input_channels, seq_len)
        
        # --- 2. Classification layers (3 Layer MLP) ---
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_classes)
        )

    def _calculate_flatten_size(self, input_channels, seq_len):
        with torch.no_grad():
            x = torch.zeros(1, input_channels, seq_len)
            x = self.features(x)
            self._to_linear = x.view(1, -1).size(1)

    def forward(self, x):
        # x shape: (Batch, Channels, Seq)
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x

def save_confusion_matrix(cm, class_names, title, filename):
    """Draw and save Normalized Confusion Matrix"""
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm[np.isnan(cm_norm)] = 0
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt=".2f", 
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Frequency'}
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Confusion matrix saved to {filename}")
    plt.close()

# --- Train and Validation ---
def train_cv_and_save_models(df):
    """
    Execute CV training, return DataFrame (with predictions), performance metrics, CM, and model parameters.
    """
    total_samples = len(df)
    best_models_dict = {} 
    
    # 1. Initialize storage for probabilities
    final_probs = np.zeros(total_samples, dtype=np.float32)
    
    fold_performance = []
    all_cv_preds = []
    all_cv_labels = []

    folds = sorted(df['fold'].unique())
    
    for fold in folds:
        print(f"\n{'='*10} Processing Fold {fold} {'='*10}")
        
        train_df = df[df['fold'] != fold]
        val_df = df[df['fold'] == fold]
        
        train_loader = DataLoader(ProteinDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
        # Note: Dataset needs to return index (batch_idx) to map back
        val_loader = DataLoader(ProteinDataset(val_df), batch_size=BATCH_SIZE, shuffle=False)
        
        model = CNNModel(num_classes=NUM_CLASSES).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) 
        
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())
        
        # --- Training Loop ---
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
            
            # Validation (Monitor Loss)
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
            
        
        # --- End of Fold: Load best model ---
        print(f" Fold {fold}: Loaded best model (Val Loss: {best_val_loss:.4f})")
        model.load_state_dict(best_model_wts)
        best_models_dict[fold] = {k: v.cpu() for k, v in best_model_wts.items()}

        # --- Calculate Metrics & Fill back probabilities ---
        model.eval()
        fold_preds, fold_labels = [], []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_idx in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                
                # Get predicted class
                preds = torch.argmax(outputs, dim=1)
                fold_preds.extend(preds.cpu().tolist())
                fold_labels.extend(batch_y.cpu().tolist())
                
                # Get predicted probabilities (Softmax) and fill final_probs
                probs = torch.nn.functional.softmax(outputs, dim=1)
                target_probs = probs.gather(1, batch_y.view(-1, 1)).squeeze()
                final_probs[batch_idx.cpu().numpy()] = target_probs.cpu().numpy()
        
        acc = accuracy_score(fold_labels, fold_preds)
        p, r, f1, _ = precision_recall_fscore_support(fold_labels, fold_preds, average='macro', zero_division=0)

        fold_performance.append({
            'Fold': fold, 'Loss': best_val_loss, 'Accuracy': acc, 
            'Precision': p, 'Recall': r, 'F1-Score': f1
        })
        all_cv_preds.extend(fold_preds)
        all_cv_labels.extend(fold_labels)

    # --- Output Summary ---
    performance_df = pd.DataFrame(fold_performance)
    avg_performance = performance_df.drop(columns=['Fold']).mean()
    print("\n" + "="*50)
    print(" Training Complete. Average Performance ")
    print(avg_performance.to_string(float_format='%.4f'))
    print("="*50)
    
    # 2. Calculate confusion matrix
    cm = confusion_matrix(all_cv_labels, all_cv_preds)
    
    # 3. Add probabilities to df (optional, but good for analysis)
    # Note: final_probs contains the probability of the *true* class
    df['prob_true_class'] = final_probs
    
    return df, performance_df, cm, best_models_dict

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}...")
        # Seq
        consensus_df = pd.read_csv(DATA_PATH)

        # Process data
        model_data_df = consensus_df[consensus_df['is_core_sample'] == True].copy()
        model_data_df = model_data_df[["species", "locus", "seq", "cluster", "silhouette_score"]]
        model_data_df.reset_index(drop=True, inplace=True) # Reset index is crucial for mapping back

        print("Generating Folds...")
        df_cross_val = make_5fold_by_gene(model_data_df)
        
        print("Starting training...")
        df_result, performance_df, cm, saved_models = train_cv_and_save_models(df_cross_val)
        
        # Save model
        torch.save(saved_models, MODEL_SAVE_PATH)
        print(f"Models saved to {MODEL_SAVE_PATH}")

        # Save result CSV
        df_result.to_csv(RESULT_CSV_PATH, index=False)
        print(f"Result DataFrame saved to {RESULT_CSV_PATH}")

        # Save Confusion Matrix
        save_confusion_matrix(
            cm, 
            class_names=CLASS_NAMES, 
            title='5-Fold CV Confusion Matrix (Normalized)',
            filename=CM_SAVE_PATH
        )
        
    else:
        print(f"Error: Data file not found at {DATA_PATH}")