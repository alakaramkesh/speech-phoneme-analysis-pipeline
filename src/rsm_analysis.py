import os
import numpy as np
import pandas as pd
import yaml

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from skbio.stats.distance import mantel


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")

with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)

ACOUSTIC_CSV = os.path.join(BASE_DIR, params["rsm_analysis"]["acoustic_csv"])
WHISPER_FILE = os.path.join(BASE_DIR, params["rsm_analysis"]["whisper_file"])
XLSR_FILE = os.path.join(BASE_DIR, params["rsm_analysis"]["xlsr_file"])
OUTPUT_CSV = os.path.join(BASE_DIR, "results/rsm_statistics.csv")

def compute_acoustic_rsm(df):
    # Keep only normalized formants
    acoustic_vectors = df[["F1_norm", "F2_norm"]].values
    # Compute pairwise Euclidean distance matrix
    distances = squareform(
        pdist(acoustic_vectors, metric="euclidean")
    )
    return distances

def compute_neural_rsm(embeddings):
    similarity_matrix = cosine_similarity(embeddings) # Compute cosine similarity matrix
    distance_matrix = 1 - similarity_matrix # Convert cosine similarity to cosine distance
    np.fill_diagonal(distance_matrix, 0)    # Force diagonal values to zero
    return distance_matrix


def main():
    # Load acoustic data
    acoustic_df = pd.read_csv(ACOUSTIC_CSV)
    # Keep only vowels
    selected_vowels = params["acoustics"]["vowels"]
    acoustic_df = acoustic_df[
        acoustic_df["label"].isin(selected_vowels)
    ].reset_index(drop=True)
    # Load Whisper embeddings
    whisper_data = np.load(WHISPER_FILE, allow_pickle=True)
    whisper_embeddings = whisper_data["embeddings"]
    whisper_labels = whisper_data["labels"]
    # Keep only vowels
    whisper_mask = np.isin(whisper_labels,selected_vowels)
    whisper_embeddings = whisper_embeddings[whisper_mask]
    # Load XLS-R embeddings
    xlsr_data = np.load(XLSR_FILE, allow_pickle=True)
    xlsr_embeddings = xlsr_data["embeddings"]
    xlsr_labels = xlsr_data["labels"]
    # Keep only vowels
    xlsr_mask = np.isin(xlsr_labels,selected_vowels)
    xlsr_embeddings = xlsr_embeddings[xlsr_mask]
    # Match minimum size across representations
    min_size = min(
        len(acoustic_df),
        len(whisper_embeddings),
        len(xlsr_embeddings)
    )
    acoustic_df = acoustic_df.iloc[:min_size]
    whisper_embeddings = whisper_embeddings[:min_size]
    xlsr_embeddings = xlsr_embeddings[:min_size]
    print(f"Using {min_size} aligned tokens")
    # Compute RSMs
    acoustic_rsm = compute_acoustic_rsm(acoustic_df)
    whisper_rsm = compute_neural_rsm(whisper_embeddings)
    xlsr_rsm = compute_neural_rsm(xlsr_embeddings)
    # Mantel tests
    whisper_corr, whisper_p, _ = mantel(
        acoustic_rsm,
        whisper_rsm,
        method="spearman"
    )
    xlsr_corr, xlsr_p, _ = mantel(
        acoustic_rsm,
        xlsr_rsm,
        method="spearman"
    )
    # Save results
    results_df = pd.DataFrame({
        "comparison": [
            "Acoustic vs Whisper",
            "Acoustic vs XLS-R"
        ],
        "mantel_correlation": [
            whisper_corr,
            xlsr_corr
        ],
        "p_value": [
            whisper_p,
            xlsr_p
        ]
    })
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(results_df)
    print(f"Saved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()