import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import umap

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")

with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)
WHISPER_FILE = os.path.join(BASE_DIR, params["neural_analysis"]["whisper_file"])
XLSR_FILE = os.path.join(BASE_DIR, params["neural_analysis"]["xlsr_file"])
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

def run_pca(embeddings):
    # Reduce embeddings to 2 dimensions using PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    return reduced

def run_umap(embeddings):
    # Reduce embeddings to 2 dimensions using UMAP
    reducer = umap.UMAP(
        n_components=2,
        random_state=42
    )
    reduced = reducer.fit_transform(embeddings)
    return reduced


def create_projection_plot(reduced, labels, title, output_path):
    # Create dataframe for plotting
    plot_df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "label": labels
    })
    plt.figure(figsize=(10, 8))
    # Plot 2D projection
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="label",
        s=15,
        alpha=0.7,
        legend=True
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to: {output_path}")
    plt.close()


def compute_between_class_variance_ratio(reduced, labels):
    # Compute global mean in 2D space
    total_mean = reduced.mean(axis=0)
    # Compute total variance
    total_variance = np.sum((reduced - total_mean) ** 2)
    between_variance = 0
    unique_labels = np.unique(labels)
    # Compute between-phoneme variance
    for label in unique_labels:
        class_points = reduced[labels == label]
        class_mean = class_points.mean(axis=0)
        between_variance += len(class_points) * np.sum(
            (class_mean - total_mean) ** 2
        )
    ratio = between_variance / total_variance
    return ratio

def compute_similarity_ratio(embeddings, labels, sample_size=3000):
    # Randomly sample a subset of tokens
    indices = np.random.choice(
        len(labels),
        size=min(sample_size, len(labels)),
        replace=False
    )
    sampled_embeddings = embeddings[indices]
    sampled_labels = labels[indices]
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(sampled_embeddings)
    within_similarities = []
    between_similarities = []
    n = len(sampled_labels)
    # Compare all sampled pairs
    for i in range(n):
        for j in range(i + 1, n):
            similarity = similarity_matrix[i, j]
            # Same phoneme pair
            if sampled_labels[i] == sampled_labels[j]:
                within_similarities.append(similarity)
            # Different phoneme pair
            else:
                between_similarities.append(similarity)
    within_mean = np.mean(within_similarities)
    between_mean = np.mean(between_similarities)
    ratio = within_mean / between_mean
    return within_mean, between_mean, ratio

def process_model(npz_path, model_name):
    print(f"\nProcessing {model_name}")
    # Load NPZ file
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    # Normalize embeddings before dimensionality reduction
    embeddings = StandardScaler().fit_transform(embeddings)
    labels = data["labels"]
    l1 = data["l1"]
    gender = data["gender"]
    selected_vowels = params["acoustics"]["vowels"]
    mask = np.isin(labels, selected_vowels)
    embeddings = embeddings[mask]
    labels = labels[mask]
    l1 = l1[mask]
    gender = gender[mask]
    # PCA projection
    pca_reduced = run_pca(embeddings)
    create_projection_plot(
        pca_reduced,
        labels,
        f"{model_name} PCA projection",
        os.path.join(FIGURE_DIR, f"{model_name.lower()}_pca.png")
    )
    pca_ratio = compute_between_class_variance_ratio(
        pca_reduced,
        labels
    )
    # UMAP projection
    umap_reduced = run_umap(embeddings)
    create_projection_plot(
        umap_reduced,
        labels,
        f"{model_name} UMAP projection",
        os.path.join(FIGURE_DIR, f"{model_name.lower()}_umap.png")
    )
    umap_ratio = compute_between_class_variance_ratio(umap_reduced,labels)
    # Cosine similarity analysis
    within_mean, between_mean, similarity_ratio = compute_similarity_ratio(embeddings,labels)
    return {
        "model": model_name,
        "pca_between_class_ratio": pca_ratio,
        "umap_between_class_ratio": umap_ratio,
        "within_similarity": within_mean,
        "between_similarity": between_mean,
        "similarity_ratio": similarity_ratio
    }

def main():
    results = []
    # Process Whisper embeddings
    whisper_results = process_model(WHISPER_FILE, "Whisper")
    results.append(whisper_results)
    # Process XLS-R embeddings
    xlsr_results = process_model(XLSR_FILE,"XLS-R")
    results.append(xlsr_results)
    # Save statistics
    results_df = pd.DataFrame(results)
    output_csv = os.path.join(OUTPUT_DIR,"neural_statistics.csv")
    results_df.to_csv(output_csv, index=False)
    print(f"Saved statistics to: {output_csv}")

if __name__ == "__main__":
    main()