import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import umap
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import os

# ==========================================
# Configuration
# ==========================================
INPUT_CSV = "./consensus_labels.csv"
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
SEQ_LENGTH = 70
PLOT_CLUSTER_FILENAME = "result_fig/umap_by_cluster.png"
PLOT_SPECIES_FILENAME = "result_fig/umap_by_species.png"
RANDOM_STATE = 42

# UMAP config
N_NEIGHBORS = 50
MIN_DIST = 0.01
METRIC = 'cosine'


def load_model(model_name, device):
    """
    Load the Pre-trained Language Model (PLM) and tokenizer.
    """
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model

def get_embeddings(sequences, tokenizer, model, device):
    """
    Extract CLS token embeddings for a list of sequences.
    """
    all_vecs = []
    print("Extracting embeddings...")
    
    with torch.no_grad():
        for seq in tqdm(sequences):
            inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            # Use the CLS token (first token)
            cls_vec = outputs.last_hidden_state[0, 0]           
            all_vecs.append(cls_vec.cpu().numpy())               

    return np.stack(all_vecs)

def perform_dimensionality_reduction(X, random_state=42):
    """
    Perform PCA followed by UMAP.
    """
    print("Running PCA...")
    pca = PCA(n_components=50, random_state=random_state)
    X_pca = pca.fit_transform(X)

    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, metric=METRIC, random_state=random_state)
    embedding_2d = reducer.fit_transform(X_pca)
    
    return embedding_2d

def plot_umap_by_cluster(embedding_2d, df, output_filename, palette_name="Set2"):
    """
    Plot UMAP results colored by cluster labels.
    """
    # Determine number of clusters dynamically
    unique_clusters = sorted(df['cluster'].unique())
    k = len(unique_clusters)
    palette = sns.color_palette(palette_name, k)
    
    # Map label to color
    # Note: Assuming cluster labels are integers 0..k-1
    # If they are not continuous integers, we might need a mapping dictionary.
    
    plt.figure(figsize=(6.5, 6), dpi=350)
    
    # Create colors list for scatter plot
    colors = [palette[label] for label in df['cluster']]
    
    plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=colors,
        s=0.5
    )

    # ----- Build legend -----
    handles = [
        mpatches.Patch(color=palette[i], label=f"Cluster {c}")
        for i, c in enumerate(unique_clusters)
    ]
    plt.legend(handles=handles, title="Clusters", loc="upper right")

    plt.title(f"Consensus cluster result with {k} subclusters")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    
    print(f"Saving cluster plot to {output_filename}...")
    plt.savefig(output_filename)
    plt.close()

def plot_umap_by_species(embedding_2d, df, output_filename, palette_name="Set3"):
    """
    Plot UMAP results colored by species.
    """
    species_list = sorted(df['species'].unique())
    n_species = len(species_list)
    
    # Generate palette (using hls if Set3 runs out of colors, otherwise Set3)
    if n_species > 12:
        palette = sns.color_palette("hls", n_species)
    else:
        palette = sns.color_palette(palette_name, n_species)

    species2color = dict(zip(species_list, palette))

    plt.figure(figsize=(6.5, 6), dpi=350)
    plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=[species2color[s] for s in df['species']],  # Color by species
        s=1
    )

    handles = [
        mpatches.Patch(color=palette[i], label=f"{species_list[i]}")
        for i in range(len(species_list))
    ]
    # Smaller fontsize for species legend as there might be many
    plt.legend(handles=handles, title="Species", fontsize=6, loc="best")
    
    plt.title("UMAP colored by species")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    
    print(f"Saving species plot to {output_filename}...")
    plt.savefig(output_filename)
    plt.close()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Check if input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Please run the clustering script first.")
        exit(1)

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    tokenizer, model = load_model(MODEL_NAME, device)

    # Load Data
    print(f"Reading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Truncate sequences
    seqs = df["seq"].str[:SEQ_LENGTH]
    
    # Extract Embeddings
    X = get_embeddings(seqs, tokenizer, model, device)
    
    # Dimensionality Reduction
    embedding_2d = perform_dimensionality_reduction(X, random_state=RANDOM_STATE)
    
    # Plotting
    plot_umap_by_cluster(embedding_2d, df, PLOT_CLUSTER_FILENAME)
    plot_umap_by_species(embedding_2d, df, PLOT_SPECIES_FILENAME)
    
    print("Done.")