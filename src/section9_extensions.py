import os
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

# paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")
with open(PARAMS_PATH, "r") as f: params = yaml.safe_load(f)

ACOUSTIC_CSV = os.path.join(BASE_DIR, params["normalization"]["output_csv"])
WHISPER_FILE = os.path.join(BASE_DIR, params["neural_analysis"]["whisper_file"])
XLSR_FILE = os.path.join(BASE_DIR, params["neural_analysis"]["xlsr_file"])
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

# phonemes
VOWELS = ["a","e","i","o","u","y","ɑ","ɛ","ø"]
CONSONANTS = params["acoustics"]["fricatives"]
ALL_PHONEMES = VOWELS + CONSONANTS

# consonant/vowel labels
PHONEME_TYPE_MAP = {}
for vowel in VOWELS: PHONEME_TYPE_MAP[vowel] = "vowel"
for consonant in CONSONANTS: PHONEME_TYPE_MAP[consonant] = "consonant"

# acoustic vectors
def load_acoustic_cv_vectors():
    df = pd.read_csv(ACOUSTIC_CSV)
    df = df[df["label"].isin(ALL_PHONEMES)].copy()
    rows = []
    for phoneme in ALL_PHONEMES:
        phoneme_df = df[df["label"] == phoneme]
        if phoneme_df.empty: continue
        if phoneme in VOWELS:
            phoneme_df = phoneme_df.dropna(subset=["F1_norm","F2_norm"])
            vector = phoneme_df[["F1_norm","F2_norm"]].mean().values
        else:
            phoneme_df = phoneme_df.dropna(subset=["duration","spectral_centroid"])
            vector = phoneme_df[["duration","spectral_centroid"]].mean().values
        rows.append({"phoneme":phoneme,"vector":vector})
    return pd.DataFrame(rows)

# neural vectors
def load_neural_cv_vectors(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    mask = np.isin(labels, ALL_PHONEMES)
    embeddings = embeddings[mask]
    labels = labels[mask]
    embeddings = StandardScaler().fit_transform(embeddings)
    embeddings = PCA(n_components=5).fit_transform(embeddings)
    rows = []
    for phoneme in ALL_PHONEMES:
        phoneme_embeddings = embeddings[labels == phoneme]
        if len(phoneme_embeddings) == 0: continue
        rows.append({"phoneme":phoneme,"vector":phoneme_embeddings.mean(axis=0)})
    return pd.DataFrame(rows)

# clustering
def run_clustering(vectors, metric, n_clusters):
    X = np.vstack(vectors["vector"])
    if metric == "euclidean":
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
    else:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    return clustering.fit_predict(X)

# consonant/vowel ARI
def compute_cv_ari(vectors, metric):
    phonemes = vectors["phoneme"].tolist()
    true_labels = [PHONEME_TYPE_MAP[p] for p in phonemes]
    predicted_clusters = run_clustering(vectors, metric, 2)
    return adjusted_rand_score(true_labels, predicted_clusters)

# silhouette analysis
def compute_silhouette_scores(vectors, metric, k_values=[2,3,4,5]):
    X = np.vstack(vectors["vector"])
    rows = []
    for k in k_values:
        if metric == "euclidean":
            clustering = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
        else:
            clustering = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        labels = clustering.fit_predict(X)
        if len(np.unique(labels)) < 2: continue
        score = silhouette_score(X, labels, metric=metric)
        rows.append({"k":k,"silhouette_score":score})
    return pd.DataFrame(rows)

def main():
    cv_results = []

    acoustic_cv_vectors = load_acoustic_cv_vectors()
    acoustic_cv_ari = compute_cv_ari(acoustic_cv_vectors, "euclidean")
    cv_results.append({
        "representation":"acoustic",
        "cv_ari":acoustic_cv_ari
    })

    whisper_cv_vectors = load_neural_cv_vectors(WHISPER_FILE)
    whisper_cv_ari = compute_cv_ari(whisper_cv_vectors, "cosine")
    cv_results.append({
        "representation":"whisper",
        "cv_ari":whisper_cv_ari
    })

    xlsr_cv_vectors = load_neural_cv_vectors(XLSR_FILE)
    xlsr_cv_ari = compute_cv_ari(xlsr_cv_vectors, "cosine")
    cv_results.append({
        "representation":"xlsr",
        "cv_ari":xlsr_cv_ari
    })

    cv_results_df = pd.DataFrame(cv_results)
    cv_output_csv = os.path.join(OUTPUT_DIR, "section9_consonant_vowel_clustering.csv")
    cv_results_df.to_csv(cv_output_csv, index=False)

    print("\nConsonant vs vowel clustering")
    print(cv_results_df)

    silhouette_results = []

    acoustic_silhouette = compute_silhouette_scores(acoustic_cv_vectors, "euclidean")
    acoustic_silhouette["representation"] = "acoustic"
    silhouette_results.append(acoustic_silhouette)

    whisper_silhouette = compute_silhouette_scores(whisper_cv_vectors, "cosine")
    whisper_silhouette["representation"] = "whisper"
    silhouette_results.append(whisper_silhouette)

    xlsr_silhouette = compute_silhouette_scores(xlsr_cv_vectors, "cosine")
    xlsr_silhouette["representation"] = "xlsr"
    silhouette_results.append(xlsr_silhouette)

    silhouette_df = pd.concat(silhouette_results, ignore_index=True)
    silhouette_output_csv = os.path.join(OUTPUT_DIR, "section9_silhouette_scores.csv")
    silhouette_df.to_csv(silhouette_output_csv, index=False)

    print("\nSilhouette scores")
    print(silhouette_df)

if __name__ == "__main__":
    main()