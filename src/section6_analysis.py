import os
import pandas as pd
import numpy as np
import yaml
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist
from scipy.stats import (
    shapiro,
    levene,
    ttest_ind,
    mannwhitneyu
)
from statsmodels.stats.multitest import multipletests
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_similarity
from skbio.stats.distance import mantel
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")
with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)
INPUT_CSV = os.path.join(BASE_DIR,params["normalization"]["output_csv"])
OUTPUT_CSV = os.path.join(BASE_DIR,"results/acoustic_tests.csv")
VOWELS = params["acoustics"]["vowels"]
WHISPER_FILE = os.path.join(BASE_DIR,params["neural_analysis"]["whisper_file"])
XLSR_FILE = os.path.join(BASE_DIR,params["neural_analysis"]["xlsr_file"])
NEURAL_OUTPUT_CSV = os.path.join(BASE_DIR,"results/neural_tests.csv")
DISTANCE_OUTPUT_DIR = os.path.join(BASE_DIR, "results/distances")


def run_statistical_test(l1_values, l2_values):
    # check normality
    shapiro_l1_p = shapiro(l1_values).pvalue
    shapiro_l2_p = shapiro(l2_values).pvalue
    # check equal variance
    levene_p = levene(l1_values, l2_values).pvalue
    normal = (
        shapiro_l1_p > 0.05 and
        shapiro_l2_p > 0.05
    )
    equal_var = levene_p > 0.05
    # choose statistical test
    if normal and equal_var:
        test_name = "t-test"
        stat, p_value = ttest_ind(
            l1_values,
            l2_values,
            equal_var=True
        )
    else:
        test_name = "mannwhitney"
        stat, p_value = mannwhitneyu(
            l1_values,
            l2_values,
            alternative="two-sided"
        )

    return {
        "test_used": test_name,
        "statistic": stat,
        "p_value": p_value,
        "shapiro_L1_p": shapiro_l1_p,
        "shapiro_L2_p": shapiro_l2_p,
        "levene_p": levene_p,
        "normal_distribution": normal,
        "equal_variance": equal_var
    }



def permutation_test(l1_embeddings, l2_embeddings, B=1000): 
    # observed centroid distance
    l1_centroid = np.mean(l1_embeddings, axis=0)
    l2_centroid = np.mean(l2_embeddings, axis=0)
    observed_distance = cosine(l1_centroid,l2_centroid)
    # combine embeddings
    combined = np.vstack([l1_embeddings,l2_embeddings])
    labels = np.array(["L1"]*len(l1_embeddings) + ["L2"]*len(l2_embeddings))
    null_distances = []
    # permutation loop
    for _ in range(B):
        shuffled = np.random.permutation(labels)
        perm_l1 = combined[shuffled == "L1"]
        perm_l2 = combined[shuffled == "L2"]
        perm_distance = cosine(
            np.mean(perm_l1, axis=0),
            np.mean(perm_l2, axis=0)
        )
        null_distances.append(perm_distance)
    null_distances = np.array(null_distances)
    # permutation p-value
    p_value = np.mean(null_distances >= observed_distance)
    return observed_distance, p_value

def run_neural_tests(npz_path, model_name):
    data = np.load(npz_path)
    embeddings = data["embeddings"]
    phonemes = data["labels"]
    groups = data["l1"]
    results = []
    for vowel in VOWELS:
        vowel_mask = phonemes == vowel
        vowel_embeddings = embeddings[vowel_mask]
        vowel_groups = groups[vowel_mask]
        l1_embeddings = vowel_embeddings[vowel_groups == "L1"]
        l2_embeddings = vowel_embeddings[vowel_groups == "L2"]
        # skip empty groups
        if (len(l1_embeddings) == 0 or len(l2_embeddings) == 0):
            continue
        distance, p_value = permutation_test(l1_embeddings,l2_embeddings)
        results.append({
            "model": model_name,
            "vowel": vowel,
            "n_L1": len(l1_embeddings),
            "n_L2": len(l2_embeddings),
            "cosine_distance": distance,
            "p_value": p_value
        })
    results_df = pd.DataFrame(results)
    # apply BH correction
    reject, corrected_p, _, _ = multipletests(
        results_df["p_value"],
        method="fdr_bh"
    )
    results_df["p_corrected"] = corrected_p
    results_df["significant"] = reject
    return results_df

def format_pvalues(df):
    pvalue_columns = [
        col for col in df.columns
        if "p" in col.lower()
    ]
    for col in pvalue_columns:
        df[col] = df[col].apply(
            lambda p:
            "<1e-6"
            if isinstance(p, (float, np.floating)) and p < 1e-6
            else "{:.3e}".format(p)
            if isinstance(p, (float, np.floating))
            else p
        )
    return df

# --------------------------------
# INTER-PHONEME DISTANCES
# --------------------------------
# compute centroid for each vowel
def compute_acoustic_centroids(df):
    centroids = []
    for vowel in VOWELS:
        vowel_df = df[df["label"] == vowel]
        centroid = vowel_df[["F1_norm", "F2_norm"]].mean().values
        centroids.append(centroid)
    return np.array(centroids)


# compute centroid embeddings
def compute_neural_centroids(embeddings, labels):
    centroids = []
    valid_vowels = []
    for vowel in VOWELS:
        vowel_embeddings = embeddings[labels == vowel]
        if len(vowel_embeddings) == 0:
            continue
        centroids.append(vowel_embeddings.mean(axis=0))
        valid_vowels.append(vowel)
    return np.array(centroids), valid_vowels

# acoustic euclidean distance matrix
def compute_acoustic_euclidean_matrix(centroids):
    return squareform(pdist(centroids, metric="euclidean"))

# acoustic mahalanobis distance matrix
def compute_acoustic_mahalanobis_matrix(centroids, acoustic_values):
    covariance = np.cov(acoustic_values, rowvar=False)
    inverse_covariance = np.linalg.pinv(covariance)
    return cdist(
        centroids,
        centroids,
        metric="mahalanobis",
        VI=inverse_covariance
    )

# neural cosine distance matrix
def compute_neural_distance_matrix(centroids):
    similarity_matrix = cosine_similarity(centroids)
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix


# save matrix heatmap
def save_heatmap(matrix, title, output_path, labels=VOWELS):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix)
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(label="distance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# save distance matrix as csv
def save_distance_csv(matrix, output_path, labels=VOWELS):
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_df.to_csv(output_path)

    # helper for shared-vowel Mantel
def run_mantel(matrix1, labels1, matrix2, labels2):
    shared = [v for v in labels1 if v in labels2]
    idx1 = [labels1.index(v) for v in shared]
    idx2 = [labels2.index(v) for v in shared]
    matrix1 = matrix1[np.ix_(idx1, idx1)]
    matrix2 = matrix2[np.ix_(idx2, idx2)]
    r, p, _ = mantel(matrix1, matrix2, method="spearman")
    return r, p


# run full inter-phoneme analysis
def run_inter_phoneme_analysis():
    os.makedirs(DISTANCE_OUTPUT_DIR, exist_ok=True)
    # load acoustic data
    acoustic_df = pd.read_csv(INPUT_CSV)
    acoustic_df = acoustic_df[
        acoustic_df["label"].isin(VOWELS)
    ].copy()
    # compute acoustic centroids
    acoustic_centroids = compute_acoustic_centroids(acoustic_df)
    acoustic_values = acoustic_df[
        ["F1_norm", "F2_norm"]
    ].dropna().values
    # acoustic distance matrices
    d_acoustic_euclidean = compute_acoustic_euclidean_matrix(acoustic_centroids)
    d_acoustic_mahalanobis = compute_acoustic_mahalanobis_matrix(acoustic_centroids, acoustic_values)
    # load whisper embeddings
    whisper_data = np.load(WHISPER_FILE, allow_pickle=True)
    whisper_embeddings = whisper_data["embeddings"]
    whisper_labels = whisper_data["labels"]
    whisper_mask = np.isin(whisper_labels, VOWELS)
    whisper_embeddings = whisper_embeddings[whisper_mask]
    whisper_labels = whisper_labels[whisper_mask]
    # whisper centroids
    whisper_centroids, whisper_vowels = compute_neural_centroids(whisper_embeddings, whisper_labels)
    d_whisper = compute_neural_distance_matrix(whisper_centroids)
    # load xlsr embeddings
    xlsr_data = np.load(XLSR_FILE, allow_pickle=True)
    xlsr_embeddings = xlsr_data["embeddings"]
    xlsr_labels = xlsr_data["labels"]
    xlsr_mask = np.isin(xlsr_labels, VOWELS)
    xlsr_embeddings = xlsr_embeddings[xlsr_mask]
    xlsr_labels = xlsr_labels[xlsr_mask]
    # xlsr centroids
    xlsr_centroids, xlsr_vowels = compute_neural_centroids(xlsr_embeddings, xlsr_labels)
    d_xlsr = compute_neural_distance_matrix(xlsr_centroids)
    # save matrices
    save_distance_csv(d_acoustic_euclidean, os.path.join(DISTANCE_OUTPUT_DIR, "Dac_euclidean.csv"))
    save_distance_csv(d_acoustic_mahalanobis, os.path.join(DISTANCE_OUTPUT_DIR, "Dac_mahalanobis.csv"))
    save_distance_csv(d_whisper, os.path.join(DISTANCE_OUTPUT_DIR, "DWh.csv"),whisper_vowels)
    save_distance_csv(d_xlsr, os.path.join(DISTANCE_OUTPUT_DIR, "DXL.csv"),xlsr_vowels)
    # save heatmaps
    save_heatmap(d_acoustic_euclidean, "Acoustic Euclidean Distance", os.path.join(DISTANCE_OUTPUT_DIR, "Dac_euclidean.png"))
    save_heatmap(d_acoustic_mahalanobis, "Acoustic Mahalanobis Distance", os.path.join(DISTANCE_OUTPUT_DIR, "Dac_mahalanobis.png"))
    save_heatmap(d_whisper, "Whisper Cosine Distance", os.path.join(DISTANCE_OUTPUT_DIR, "DWh.png"),whisper_vowels)
    save_heatmap(d_xlsr, "XLS-R Cosine Distance", os.path.join(DISTANCE_OUTPUT_DIR, "DXL.png"),xlsr_vowels)
    # mantel tests
    acoustic_whisper_r, acoustic_whisper_p = run_mantel(d_acoustic_euclidean,VOWELS,d_whisper,whisper_vowels)
    acoustic_xlsr_r, acoustic_xlsr_p = run_mantel(d_acoustic_euclidean,VOWELS,d_xlsr,xlsr_vowels)
    whisper_xlsr_r, whisper_xlsr_p = run_mantel(d_whisper,whisper_vowels,d_xlsr,xlsr_vowels)
    # save mantel results
    mantel_results = pd.DataFrame({
        "comparison": [
            "Dac vs DWh",
            "Dac vs DXL",
            "DWh vs DXL"
        ],
        "mantel_r": [
            acoustic_whisper_r,
            acoustic_xlsr_r,
            whisper_xlsr_r
        ],
        "p_value": [
            acoustic_whisper_p,
            acoustic_xlsr_p,
            whisper_xlsr_p
        ]
    })
    mantel_results = format_pvalues(mantel_results)
    mantel_results.to_csv(
        os.path.join(DISTANCE_OUTPUT_DIR, "mantel_results.csv"),
        index=False
    )
    print(mantel_results)
    print(f"Saved inter-phoneme distance analysis to: {DISTANCE_OUTPUT_DIR}")
    # bootstrap confidence intervals
    bootstrap_results = []
    phoneme_pairs = [
        ("e", "ɛ"),
        ("y", "u")
    ]
    for vowel1, vowel2 in phoneme_pairs:
        # acoustic distance
        acoustic_distance = d_acoustic_euclidean[
            VOWELS.index(vowel1),
            VOWELS.index(vowel2)
        ]
        acoustic_ci = bootstrap_ci([acoustic_distance] * 100)
        # whisper distance
        whisper_distance = d_whisper[
            VOWELS.index(vowel1),
            VOWELS.index(vowel2)
        ]
        whisper_ci = bootstrap_ci([whisper_distance] * 100)
        # xlsr distance
        xlsr_distance = d_xlsr[
            VOWELS.index(vowel1),
            VOWELS.index(vowel2)
        ]
        xlsr_ci = bootstrap_ci([xlsr_distance] * 100)
        bootstrap_results.extend([
            {
                "representation": "acoustic",
                "pair": f"{vowel1}-{vowel2}",
                "distance": acoustic_distance,
                "ci_lower": acoustic_ci[0],
                "ci_upper": acoustic_ci[1]
            },
            {
                "representation": "whisper",
                "pair": f"{vowel1}-{vowel2}",
                "distance": whisper_distance,
                "ci_lower": whisper_ci[0],
                "ci_upper": whisper_ci[1]
            },
            {
                "representation": "xlsr",
                "pair": f"{vowel1}-{vowel2}",
                "distance": xlsr_distance,
                "ci_lower": xlsr_ci[0],
                "ci_upper": xlsr_ci[1]
            }
        ])
    bootstrap_df = pd.DataFrame(
        bootstrap_results
    )
    bootstrap_df.to_csv(
        os.path.join(
            DISTANCE_OUTPUT_DIR,
            "bootstrap_results.csv"
        ),
        index=False
    )
    print(bootstrap_df)

# bootstrap confidence interval
def bootstrap_ci(values, n_bootstrap=1000):
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values,size=len(values),replace=True)
        bootstrap_means.append(np.mean(sample))
    lower_ci = np.percentile(bootstrap_means,2.5)
    upper_ci = np.percentile(bootstrap_means,97.5)
    return lower_ci, upper_ci

# --------------------------------
# SIMPLE PHONEME IDENTIFICATION
# --------------------------------
def evaluate_classifier(features, labels, speakers):
    predictions = []
    true_labels = []
    for speaker in np.unique(speakers):
        train_mask = speakers != speaker
        test_mask = speakers == speaker
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(features[train_mask], labels[train_mask])
        y_pred = clf.predict(features[test_mask])
        predictions.extend(y_pred)
        true_labels.extend(labels[test_mask])
    return accuracy_score(true_labels, predictions), np.array(predictions), np.array(true_labels)

def run_simple_classifier_analysis():
    acoustic_df = pd.read_csv(INPUT_CSV)
    acoustic_df = acoustic_df[acoustic_df["label"].isin(VOWELS)].copy()
    acoustic_acc, acoustic_pred, acoustic_true = evaluate_classifier(
        acoustic_df[["F1_norm", "F2_norm"]].values,
        acoustic_df["label"].values,
        acoustic_df["speaker_id"].values
    )
    whisper_data = np.load(WHISPER_FILE, allow_pickle=True)
    whisper_mask = np.isin(whisper_data["labels"], VOWELS)
    whisper_acc,_,_ = evaluate_classifier(
        whisper_data["embeddings"][whisper_mask],
        whisper_data["labels"][whisper_mask],
        whisper_data["speakers"][whisper_mask]
    )
    xlsr_data = np.load(XLSR_FILE, allow_pickle=True)
    xlsr_mask = np.isin(xlsr_data["labels"], VOWELS)
    xlsr_acc,_,_ = evaluate_classifier(
        xlsr_data["embeddings"][xlsr_mask],
        xlsr_data["labels"][xlsr_mask],
        xlsr_data["speakers"][xlsr_mask]
    )
    results_df = pd.DataFrame({
        "representation": ["acoustic", "whisper", "xlsr"],
        "accuracy": [acoustic_acc, whisper_acc, xlsr_acc]
    })
    output_csv = os.path.join(BASE_DIR, "results/classification_results.csv")
    results_df.to_csv(output_csv, index=False)
    print(results_df)
    print(f"Saved classification results to: {output_csv}")

def main():
    df = pd.read_csv(INPUT_CSV)
    # keep only vowels
    vowel_df = df[df["label"].isin(VOWELS)].copy()
    print(f"Vowel rows: {len(vowel_df)}")
    results = []
    # test each vowel separately
    for vowel in VOWELS:
        vowel_subset = vowel_df[
            vowel_df["label"] == vowel
        ]
        for feature in ["F1_norm", "F2_norm"]:
            l1_values = vowel_subset[vowel_subset["L1"] == "L1"][feature].dropna()
            l2_values = vowel_subset[vowel_subset["L1"] == "L2"][feature].dropna()
            # skip empty groups
            if len(l1_values) == 0 or len(l2_values) == 0:
                continue
            stats_result = run_statistical_test(l1_values,l2_values)
            results.append({
                "vowel": vowel,
                "feature": feature,
                "n_L1": len(l1_values),
                "n_L2": len(l2_values),
                "mean_L1": np.mean(l1_values),
                "mean_L2": np.mean(l2_values),
                "test_used": stats_result["test_used"],
                "statistic": stats_result["statistic"],
                "p_value": stats_result["p_value"],
                "shapiro_L1_p": stats_result["shapiro_L1_p"],
                "shapiro_L2_p": stats_result["shapiro_L2_p"],
                "levene_p": stats_result["levene_p"],
                "normal_distribution": stats_result["normal_distribution"],
                "equal_variance": stats_result["equal_variance"],
            })
    results_df = pd.DataFrame(results)
    # apply FDR correction
    reject, corrected_p, _, _ = multipletests(
        results_df["p_value"],
        method="fdr_bh"
    )
    results_df["p_corrected"] = corrected_p
    results_df["significant"] = reject
    results_df = format_pvalues(results_df)
    # neural tests
    whisper_results = run_neural_tests(WHISPER_FILE, "whisper")
    xlsr_results = run_neural_tests(XLSR_FILE, "xlsr")
    neural_results = pd.concat([whisper_results, xlsr_results])
    neural_results = format_pvalues(neural_results)
    # save results
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    neural_results.to_csv(NEURAL_OUTPUT_CSV, index=False)
    print(results_df)
    print(neural_results)
    print(f"Saved acoustic results to: {OUTPUT_CSV}")
    print(f"Saved neural results to: {NEURAL_OUTPUT_CSV}")
    run_inter_phoneme_analysis()
    run_simple_classifier_analysis()

if __name__ == "__main__":
    main()