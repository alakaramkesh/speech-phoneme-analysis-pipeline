import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import pdist

# paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")
with open(PARAMS_PATH, "r") as f: params = yaml.safe_load(f)
ACOUSTIC_CSV = os.path.join(BASE_DIR, params["normalization"]["output_csv"])
WHISPER_FILE = os.path.join(BASE_DIR, params["neural_analysis"]["whisper_file"])
XLSR_FILE = os.path.join(BASE_DIR, params["neural_analysis"]["xlsr_file"])
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "results/figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
# keep only vowels needed for clustering evaluation
# front/back labels
FRONT_BACK_MAP = {
    "i":"front",
    "y":"front",
    "e":"front",
    "ø":"front",
    "ɛ":"front",
    "a":"front",
    "u":"back",
    "o":"back",
    "ɔ":"back",
    "ɑ":"back"
}
# vowel height labels
HEIGHT_MAP = {
    "i":"high",
    "y":"high",
    "u":"high",
    "e":"mid",
    "ø":"mid",
    "o":"mid",
    "ɛ":"low",
    "ɔ":"low",
    "a":"low",
    "ɑ":"low"
}
VOWELS = [v for v in params["acoustics"]["vowels"] if v in FRONT_BACK_MAP and v in HEIGHT_MAP]
# acoustic vectors
def load_acoustic_vectors():
    df = pd.read_csv(ACOUSTIC_CSV)
    df = df[df["label"].isin(VOWELS)].copy()
    df = df.dropna(subset=["F1_norm","F2_norm"])
    rows = []
    for vowel in VOWELS:
        vowel_df = df[df["label"] == vowel]
        if vowel_df.empty: continue
        rows.append({"phoneme": vowel,"vector": vowel_df[["F1_norm","F2_norm"]].mean().values})
    return pd.DataFrame(rows)
# neural vectors
def load_neural_vectors(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    mask = np.isin(labels, VOWELS)
    embeddings = embeddings[mask]
    labels = labels[mask]
    # normalize neural embeddings
    embeddings = StandardScaler().fit_transform(embeddings)
    # lightweight PCA for faster clustering
    embeddings = PCA(n_components=5).fit_transform(embeddings)
    rows = []
    for vowel in VOWELS:
        vowel_embeddings = embeddings[labels == vowel]
        if len(vowel_embeddings) == 0: continue
        rows.append({"phoneme": vowel,"vector": vowel_embeddings.mean(axis=0)})
    return pd.DataFrame(rows)
# clustering
def run_clustering(vectors, metric, n_clusters):
    X = np.vstack(vectors["vector"])
    if metric == "euclidean":
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
    else:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    return clustering.fit_predict(X)
# ari scores
def compute_ari_scores(vectors, metric):
    phonemes = vectors["phoneme"].tolist()
    front_back_true = [FRONT_BACK_MAP[p] for p in phonemes]
    height_true = [HEIGHT_MAP[p] for p in phonemes]
    front_back_clusters = run_clustering(vectors, metric, 2)
    height_clusters = run_clustering(vectors, metric, 3)
    front_back_ari = adjusted_rand_score(front_back_true, front_back_clusters)
    height_ari = adjusted_rand_score(height_true, height_clusters)
    return front_back_ari, height_ari
# dendrogram
def make_dendrogram(vectors, metric, title, output_name):
    X = np.vstack(vectors["vector"])
    if metric == "euclidean":
        linkage_matrix = sch.linkage(X, method="ward")
    else:
        distance_matrix = pdist(X, metric="cosine")
        linkage_matrix = sch.linkage(distance_matrix, method="average")
    plt.figure(figsize=(8,6))
    sch.dendrogram(linkage_matrix, labels=vectors["phoneme"].tolist())
    plt.title(title)
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, output_name)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def main():
    results = []
    acoustic_vectors = load_acoustic_vectors()
    acoustic_front_back_ari, acoustic_height_ari = compute_ari_scores(acoustic_vectors, "euclidean")
    results.append({
        "representation":"acoustic",
        "front_back_ari":acoustic_front_back_ari,
        "height_ari":acoustic_height_ari
    })
    make_dendrogram(acoustic_vectors, "euclidean", "Acoustic vowel clustering", "dendrogram_acoustic.png")
    whisper_vectors = load_neural_vectors(WHISPER_FILE)
    whisper_front_back_ari, whisper_height_ari = compute_ari_scores(whisper_vectors, "cosine")
    results.append({
        "representation":"whisper",
        "front_back_ari":whisper_front_back_ari,
        "height_ari":whisper_height_ari
    })
    make_dendrogram(whisper_vectors, "cosine", "Whisper vowel clustering", "dendrogram_whisper.png")
    xlsr_vectors = load_neural_vectors(XLSR_FILE)
    xlsr_front_back_ari, xlsr_height_ari = compute_ari_scores(xlsr_vectors, "cosine")
    results.append({
        "representation":"xlsr",
        "front_back_ari":xlsr_front_back_ari,
        "height_ari":xlsr_height_ari
    })
    make_dendrogram(xlsr_vectors, "cosine", "XLS-R vowel clustering", "dendrogram_xlsr.png")
    results_df = pd.DataFrame(results)
    output_csv = os.path.join(OUTPUT_DIR, "section9_vowel_clustering.csv")
    results_df.to_csv(output_csv, index=False)
    print(results_df)

if __name__ == "__main__":
    main()