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
# SPEAKER REPRESENTATIONS
def load_acoustic_speaker_vectors():
    df = pd.read_csv(ACOUSTIC_CSV)
    df = df[df["label"].isin(VOWELS)].copy()
    df = df.dropna(subset=["F1_norm","F2_norm","speaker_id","L1","gender"])
    rows = []
    for speaker in sorted(df["speaker_id"].unique()):
        speaker_df = df[df["speaker_id"] == speaker]
        vowel_vectors = []
        for vowel in VOWELS:
            vowel_df = speaker_df[speaker_df["label"] == vowel]
            if vowel_df.empty:
                vowel_vectors.extend([0,0])
                continue
            vowel_vectors.extend(vowel_df[["F1_norm","F2_norm"]].mean().values.tolist())
        rows.append({
            "speaker_id": speaker,
            "L1": speaker_df["L1"].iloc[0],
            "gender": speaker_df["gender"].iloc[0],
            "vector": np.array(vowel_vectors)
        })
    return pd.DataFrame(rows)
def load_neural_speaker_vectors(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    speakers = data["speakers"]
    l1 = data["l1"]
    gender = data["gender"]
    mask = np.isin(labels, VOWELS)
    embeddings = embeddings[mask]
    labels = labels[mask]
    speakers = speakers[mask]
    l1 = l1[mask]
    gender = gender[mask]
    embeddings = StandardScaler().fit_transform(embeddings)
    embeddings = PCA(n_components=5).fit_transform(embeddings)
    rows = []
    for speaker in sorted(np.unique(speakers)):
        speaker_mask = speakers == speaker
        speaker_embeddings = embeddings[speaker_mask]
        speaker_labels = labels[speaker_mask]
        vowel_vectors = []
        for vowel in VOWELS:
            vowel_embeddings = speaker_embeddings[speaker_labels == vowel]
            if len(vowel_embeddings) == 0:
                vowel_vectors.extend([0] * embeddings.shape[1])
                continue
            vowel_vectors.extend(vowel_embeddings.mean(axis=0).tolist())
        rows.append({
            "speaker_id": speaker,
            "L1": l1[speaker_mask][0],
            "gender": gender[speaker_mask][0],
            "vector": np.array(vowel_vectors)
        })
    return pd.DataFrame(rows)
# SPEAKER CLUSTERING
def compute_speaker_clustering_ari(vectors, metric):
    X = np.vstack(vectors["vector"])
    if metric == "euclidean":
        clustering = AgglomerativeClustering(n_clusters=2, metric="euclidean", linkage="ward")
    else:
        clustering = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average")
    cluster_labels = clustering.fit_predict(X)
    l1_true = vectors["L1"].astype(str).tolist()
    gender_true = vectors["gender"].astype(str).tolist()
    l1_ari = adjusted_rand_score(l1_true, cluster_labels)
    gender_ari = adjusted_rand_score(gender_true, cluster_labels)
    return l1_ari, gender_ari
# SPEAKER DENDROGRAM
def make_speaker_dendrogram(vectors, metric, title, output_name):
    X = np.vstack(vectors["vector"])
    labels = vectors["speaker_id"].astype(str).tolist()
    if metric == "euclidean":
        linkage_matrix = sch.linkage(X, method="ward")
    else:
        distance_matrix = pdist(X, metric="cosine")
        linkage_matrix = sch.linkage(distance_matrix, method="average")
    plt.figure(figsize=(10,6))
    sch.dendrogram(linkage_matrix, labels=labels)
    plt.title(title)
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, output_name)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")
# PHONEME MISCLASSIFICATION
def get_cluster_labels(vectors, metric, n_clusters):
    X = np.vstack(vectors["vector"])
    if metric == "euclidean": clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
    else: clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    return clustering.fit_predict(X)
def analyse_phoneme_misclassification(vectors, metric, representation):
    phonemes = vectors["phoneme"].tolist()
    front_back_true = [FRONT_BACK_MAP[p] for p in phonemes]
    height_true = [HEIGHT_MAP[p] for p in phonemes]
    front_back_clusters = get_cluster_labels(vectors, metric, 2)
    height_clusters = get_cluster_labels(vectors, metric, 3)
    rows = []
    for i, phoneme in enumerate(phonemes):
        rows.append({"representation":representation,"phoneme":phoneme,"true_front_back":front_back_true[i],"front_back_cluster":front_back_clusters[i],"true_height":height_true[i],"height_cluster":height_clusters[i]})
    return pd.DataFrame(rows)
def find_systematic_misclassifications(misclassification_df):
    rows = []
    for phoneme in sorted(misclassification_df["phoneme"].unique()):
        phoneme_df = misclassification_df[misclassification_df["phoneme"] == phoneme]
        fb_clusters = phoneme_df["front_back_cluster"].nunique()
        height_clusters = phoneme_df["height_cluster"].nunique()
        rows.append({"phoneme":phoneme,"front_back_cluster_variation":fb_clusters,"height_cluster_variation":height_clusters,"representations_seen":len(phoneme_df)})
    return pd.DataFrame(rows)
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
    # SECTION 9.3 — speaker clustering
    speaker_results = []

    acoustic_speakers = load_acoustic_speaker_vectors()
    acoustic_l1_ari, acoustic_gender_ari = compute_speaker_clustering_ari(acoustic_speakers, "euclidean")
    speaker_results.append({
        "representation":"acoustic",
        "L1_ARI":acoustic_l1_ari,
        "gender_ARI":acoustic_gender_ari
    })
    make_speaker_dendrogram(acoustic_speakers, "euclidean", "Acoustic speaker clustering", "speaker_dendrogram_acoustic.png")

    whisper_speakers = load_neural_speaker_vectors(WHISPER_FILE)
    whisper_l1_ari, whisper_gender_ari = compute_speaker_clustering_ari(whisper_speakers, "cosine")
    speaker_results.append({
        "representation":"whisper",
        "L1_ARI":whisper_l1_ari,
        "gender_ARI":whisper_gender_ari
    })
    make_speaker_dendrogram(whisper_speakers, "cosine", "Whisper speaker clustering", "speaker_dendrogram_whisper.png")

    xlsr_speakers = load_neural_speaker_vectors(XLSR_FILE)
    xlsr_l1_ari, xlsr_gender_ari = compute_speaker_clustering_ari(xlsr_speakers, "cosine")
    speaker_results.append({
        "representation":"xlsr",
        "L1_ARI":xlsr_l1_ari,
        "gender_ARI":xlsr_gender_ari
    })
    make_speaker_dendrogram(xlsr_speakers, "cosine", "XLS-R speaker clustering", "speaker_dendrogram_xlsr.png")
    speaker_results_df = pd.DataFrame(speaker_results)
    speaker_output_csv = os.path.join(OUTPUT_DIR, "section9_speaker_clustering.csv")
    speaker_results_df.to_csv(speaker_output_csv, index=False)
    print("\nSpeaker clustering results")
    print(speaker_results_df)
    # SECTION 9.4 / Q16 — phoneme misclassification check
    misclassification_results = []
    misclassification_results.append(analyse_phoneme_misclassification(acoustic_vectors, "euclidean", "acoustic"))
    misclassification_results.append(analyse_phoneme_misclassification(whisper_vectors, "cosine", "whisper"))
    misclassification_results.append(analyse_phoneme_misclassification(xlsr_vectors, "cosine", "xlsr"))
    misclassification_df = pd.concat(misclassification_results, ignore_index=True)
    misclassification_csv = os.path.join(OUTPUT_DIR, "section9_phoneme_clusters.csv")
    misclassification_df.to_csv(misclassification_csv, index=False)
    systematic_df = find_systematic_misclassifications(misclassification_df)
    systematic_csv = os.path.join(OUTPUT_DIR, "section9_systematic_phoneme_patterns.csv")
    systematic_df.to_csv(systematic_csv, index=False)
    print("\nPhoneme cluster assignments")
    print(misclassification_df)
    print("\nSystematic phoneme patterns")
    print(systematic_df)

if __name__ == "__main__":
    main()