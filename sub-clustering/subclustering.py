import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score
import igraph as ig
import leidenalg
import pynndescent
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import itertools
from tqdm import tqdm
import warnings
import logging
import os

# --- Global config ---
INPUT_CSV = "../all_species_data.csv" # Your data
OUTPUT_CSV = "./consensus_labels.csv" # Clustering result file
MODEL_NAME = "facebook/esm2_t6_8M_UR50D" # PLM model name
SILHOUETTE_PLOT_FILENAME = "result_fig/consensus_silhouette.png"
HEATMAP_PLOT_FILENAME = "result_fig/consensus_heatmap.png"


# Setup logging and warnings
warnings.filterwarnings("ignore", message="Graph is not fully connected")
logging.getLogger("transformers").setLevel(logging.ERROR)

def extend_from_n_terminal(df, length):
    """
    Extract fixed length segments from the N-terminal.
    """
    return df["seq"].str[:length] # Your seq data column name == "seq"

def extract_es_embeddings(
    sequences,
    tokenizer,
    model,
    segment_lengths=None,
    device=None,
    show_progress=True,
):
    """
    General ESM embedding extraction (mean pooling).
    - If segment_lengths is None -> Average over the full length (removing CLS/EOS).
    - If segment_lengths is provided -> Mean pooling from CLS over the specified length.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_vecs = []
    iterator = tqdm(sequences) if show_progress else sequences

    with torch.no_grad():
        for i, seq in enumerate(iterator):
            inputs = tokenizer(
                seq, return_tensors="pt", add_special_tokens=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            hidden = outputs.last_hidden_state.squeeze(0)  # (L, D)

            # Determine the segment for mean pooling
            if segment_lengths is None:
                residue_repr = hidden[1:-1]  # full length (remove CLS/EOS)
            else:
                seg_len = segment_lengths[i]
                residue_repr = hidden[1:1 + seg_len]

            pooled = residue_repr.mean(dim=0)
            all_vecs.append(pooled.cpu().numpy())

    return np.stack(all_vecs)

def kmeans_all_cosine(X_pca, k_range=range(2, 11), random_state=42, sample_size=None, verbose=False, minibatch=False):
    X_norm = normalize(X_pca, norm="l2", axis=1)
    results = []
    for k in k_range:
        if minibatch:
            km = MiniBatchKMeans(n_clusters=k, n_init="auto", random_state=random_state)
        else:
            km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)

        labels = km.fit_predict(X_norm)
        score = silhouette_score(X_norm, labels, metric="cosine", sample_size=sample_size, random_state=random_state)
        if verbose:
            print(f"[kmeans] k={k}, silhouette={score:.3f}")
        params = {"k": k}
        results.append((labels, score, params))
    return results

def spectral_all_cosine(X_pca, k_range=range(2, 11), n_neighbors=30, random_state=42, verbose=False):
    # L2 normalize
    Xn = normalize(X_pca, norm="l2", axis=1)

    # kNN graph (distance)
    G_dist = kneighbors_graph(Xn, n_neighbors=n_neighbors, mode="distance", metric="euclidean", include_self=False)
    S = G_dist.copy()
    S.data = 1.0 - S.data
    S.data[S.data < 0] = 0.0
    S = 0.5 * (S + S.T)
    S = S + csr_matrix(np.eye(S.shape[0]) * 1e-6)

    results = []
    for k in k_range:
        sc = SpectralClustering(n_clusters=k, affinity="precomputed", assign_labels="kmeans", n_init=50, random_state=random_state,
                                eigen_solver="amg", eigen_tol=1e-3)
        labels = sc.fit_predict(S)
        if len(np.unique(labels)) < 2:
            score = -1.0
        else:
            score = silhouette_score(Xn, labels, metric="cosine")

        if verbose:
            print(f"[spectral] k={k}, silhouette(cosine)={score:.3f}")

        params = {"k": k, "n_neighbors": n_neighbors}
        results.append((labels, score, params))
    return results

def leiden_all_cosine(X_pca, n_neighbors_list=None, resolutions=None, random_state=42, verbose=False):
    if n_neighbors_list is None:
        n_neighbors_list = [10, 30, 50]
    if resolutions is None:
        resolutions = [0.2, 0.5, 0.8]

    X_norm = normalize(X_pca, norm="l2", axis=1)

    all_results = []
    for n_neighbors in n_neighbors_list:
        nnd = pynndescent.NNDescent(X_norm, n_neighbors=n_neighbors, metric="cosine", random_state=random_state, n_jobs=-1, compressed=True)
        indices, _ = nnd.neighbor_graph
        edges = []
        for i in range(indices.shape[0]):
            for j in indices[i]:
                if i < j:
                    edges.append((i, j))
        g = ig.Graph(edges=edges, directed=False)
        g = g.simplify()

        for res in resolutions:
            part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res, seed=random_state)
            labels = np.array(part.membership)
            modularity = g.modularity(labels)

            if verbose:
                print(
                    f"[leiden] n_neighbors={n_neighbors}, "
                    f"res={res}, clusters={len(set(labels))}, "
                    f"modularity={modularity:.3f}"
                )
            params = {"n_neighbors": n_neighbors, "resolution": res}
            all_results.append((labels, modularity, params))

    return all_results

def build_consensus_matrix(plastid_df, tokenizer, model, n_lengths=[50, 60, 70, 80, 90, 100], n_components=50, k_range=range(2, 11)):
    """
    Build Weighted Consensus Matrix:
    - Include all k results.
    - Calculate score using log1p(max(score, 0)).
    - Normalize within each algorithm method.
    """
    N = len(plastid_df)
    consensus_matrix = np.zeros((N, N), dtype=np.float32)
    cluster_results = []
    total_weight = 0.0

    def accumulate(labels, w, tag, params):
        nonlocal consensus_matrix, total_weight, cluster_results
        same_cluster = np.equal.outer(labels, labels).astype(np.float32)
        consensus_matrix += same_cluster * w
        total_weight += w
        cluster_results.append((labels, params, w, tag))

    def spectral_runner(X_pca):
        return spectral_all_cosine(X_pca, k_range=k_range, n_neighbors=30, random_state=42, verbose=False)

    def kmeans_runner(X_pca):
        return kmeans_all_cosine(X_pca, k_range=k_range, random_state=42, sample_size=None, verbose=False, minibatch=False)

    def leiden_runner(X_pca):
        return leiden_all_cosine(X_pca, n_neighbors_list=[10, 30, 50], resolutions=[0.2, 0.5, 0.8], random_state=42, verbose=False)

    methods = [("spectral", spectral_runner), ("kmeans", kmeans_runner), ("leiden", leiden_runner)]

    def process_condition(prefix, X_pca):
        alpha = 2.0  # Increase weight for high scores

        for method_name, func in methods:
            results = func(X_pca)  # list[(labels, score, params)]

            # score = silhouette or modularity
            safe_scores = [max(float(s), 0.0) for (_, s, _) in results]
            trans_scores = [s ** alpha for s in safe_scores]

            sum_scores = sum(trans_scores)
            if sum_scores <= 0:
                continue

            weights = [s / sum_scores for s in trans_scores]

            for (labels, score, params), w in zip(results, weights):
                accumulate(labels, w, f"{prefix}_{method_name}", params)

    for L in n_lengths:
        seqs = extend_from_n_terminal(plastid_df, L)
        X = extract_es_embeddings(seqs.tolist(), tokenizer, model, segment_lengths=None)
        X_pca = PCA(n_components=n_components, random_state=42).fit_transform(X)
        process_condition(f"Fix_{L}", X_pca)

    if total_weight > 0:
        consensus_matrix /= total_weight
    else:
        print("⚠️ total_weight == 0, all clustering scores are <= 0, returning zero consensus matrix.")

    return consensus_matrix, cluster_results

def postprocess_consensus_matrix(C):
    C = np.asarray(C, dtype=float)
    C = np.clip(C, 0.0, 1.0)
    np.fill_diagonal(C, 1.0)
    return C

def silhouette_scores_consensus_spectral(consensus_matrix, k_range=range(2, 10), random_state=42):
    """
    Cluster the consensus matrix using Spectral Clustering at different k,
    and calculate the Silhouette score.
    """
    C = np.asarray(consensus_matrix)
    assert C.ndim == 2 and C.shape[0] == C.shape[1], "Consensus matrix must be an NxN square matrix"

    sil_scores = {}

    for k in k_range:
        # Build Spectral Clustering model
        model = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",  # Or "discretize"
            random_state=random_state
        )

        try:
            labels = model.fit_predict(C)
            # Calculate silhouette only if there are at least 2 clusters
            if len(np.unique(labels)) < 2:
                sil_scores[k] = np.nan
                continue
            # Distance matrix: 1 - similarity
            D = 1 - C
            score = silhouette_score(D, labels, metric="precomputed")
            sil_scores[k] = score
        except Exception as e:
            print(f"Error at k={k}: {e}")
            sil_scores[k] = np.nan

    return sil_scores

def plot_consensus_with_groupbar_spectral_hybrid(
    C,
    n_clusters=5,
    cmap="viridis",
    palette="Set2",
    figsize=(10, 10),
    dpi=300,
    bar_position="bottom",
    random_state=42,
    linkage_method="average"
):
    """
    Hybrid method:
    Spectral clustering first -> Hierarchical clustering within groups -> Plot consensus heatmap and group bars.
    """

    # === Preprocessing ===
    C = np.asarray(C)
    assert C.ndim == 2 and C.shape[0] == C.shape[1], "C must be an NxN matrix"
    n = C.shape[0]
    C = (C + C.T) / 2.0
    np.fill_diagonal(C, 1.0)

    # === Step 1. Spectral Clustering ===
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state
    )
    labels = model.fit_predict(C)

    # === Step 2. Hierarchical sorting within groups ===
    final_order = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) > 1:
            # Hierarchical clustering on the sub-matrix (distance matrix 1-C)
            subC = C[np.ix_(idx, idx)]
            D = 1 - subC
            Z = linkage(squareform(D, checks=False), method=linkage_method)
            local_order = leaves_list(Z)
            idx_ordered = idx[local_order]
        else:
            idx_ordered = idx
        final_order.extend(idx_ordered)

    order = np.array(final_order)
    C_ord = C[np.ix_(order, order)]
    labels_ord = labels[order]

    # === Step 3. Color Setup ===
    uniq = np.sort(np.unique(labels_ord))
    pal = sns.color_palette(palette, len(uniq))
    lut = {lab: col for lab, col in zip(uniq, pal)}

    # === Step 4. Plotting ===
    fig = plt.figure(figsize=figsize, dpi=dpi, layout="constrained")

    if bar_position == "top":
        heights = [0.8, 20]
        gs = fig.add_gridspec(2, 1, height_ratios=heights)
        ax_bar = fig.add_subplot(gs[0])
        ax_heat = fig.add_subplot(gs[1], sharex=ax_bar)
    else:
        heights = [20, 0.8]
        gs = fig.add_gridspec(2, 1, height_ratios=heights)
        ax_heat = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1], sharex=ax_heat)

    # === Heatmap ===
    import seaborn as sns # Ensure seaborn is imported here if not at top level
    sns.heatmap(
        C_ord, ax=ax_heat, cmap=cmap, cbar=True,
        xticklabels=False, yticklabels=False
    )
    ax_heat.set_title(f"Consensus Matrix, k={n_clusters}", pad=12)

    # === Group Bars ===
    start = 0
    for lab, group in itertools.groupby(labels_ord):
        members = list(group)
        length = len(members)
        ax_bar.add_patch(
            plt.Rectangle((start, 0), length, 1,
                          color=lut[lab], ec="none")
        )
        ax_bar.text(start + length/2, 0.5, str(length),
                    ha="center", va="center", fontsize=8, color="black")
        start += length

    ax_bar.set_xlim(0, n)
    ax_bar.set_ylim(0, 1)
    ax_bar.axis("off")

    return labels, order, fig

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    print(f"Reading data from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: File {INPUT_CSV} not found. Please check the path.")
        exit(1)

    plastid_df = df[df['location'] == "plastid"]
    print(f"Filtered {len(plastid_df)} plastid sequences.")

    # Load PLM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Build Consensus Matrix
    print("Building consensus matrix...")
    consensus_matrix, cluster_results = build_consensus_matrix(plastid_df, tokenizer, model)

    # Post-process
    consensus_matrix = postprocess_consensus_matrix(consensus_matrix)
    print("Saving consensus matrix...")
    np.savez_compressed("consensus_matrix.npz", consensus_matrix=consensus_matrix)

    # Calculate Silhouette Scores
    print("Calculating silhouette scores for consensus matrix...")
    sil_scores = silhouette_scores_consensus_spectral(consensus_matrix, k_range=range(2, 11))

    # Plot Silhouette Scores
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette score vs k (Spectral Clustering on Consensus Matrix)")
    
    print(f"Saving silhouette plot to {SILHOUETTE_PLOT_FILENAME}...")
    plt.savefig(SILHOUETTE_PLOT_FILENAME)
    plt.close()

    # Plot Consensus Heatmap
    # Set the number of clusters for final visualization (example: 8)
    N_CLUSTERS = 8 
    print(f"Plotting consensus heatmap with {N_CLUSTERS} clusters...")
    
    # Need to import seaborn here if not already imported at top
    import seaborn as sns 
    
    consensus_labels, order, fig = plot_consensus_with_groupbar_spectral_hybrid(consensus_matrix, n_clusters=N_CLUSTERS)
    
    print(f"Saving heatmap to {HEATMAP_PLOT_FILENAME}...")
    fig.savefig(HEATMAP_PLOT_FILENAME)
    plt.close(fig)

    # Save Results
    print(f"Saving cluster labels to {OUTPUT_CSV}...")
    cluster_result_df = pd.DataFrame({
        "species": plastid_df["species"],
        "locus": plastid_df["locus"],
        "seq": plastid_df["seq"],
        "cluster": consensus_labels
    })

    cluster_result_df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")