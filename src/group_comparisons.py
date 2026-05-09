import os
import pandas as pd
import numpy as np
import yaml

from scipy.stats import (
    shapiro,
    levene,
    ttest_ind,
    mannwhitneyu
)

from statsmodels.stats.multitest import multipletests
from scipy.spatial.distance import cosine


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
NEURAL_OUTPUT_CSV = os.path.join(BASE_DIR,"results/neural_tests.csv"
)

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

def format_pvalues(df): # make report numbers prettier
    pvalue_columns = [
        col for col in df.columns
        if "p" in col.lower()
    ]
    for col in pvalue_columns:
        df[col] = df[col].apply(
            lambda p: "<1e-6" if p < 1e-6 else "{:.3e}".format(p)               
            if isinstance(p, (float, np.floating))
            else p
        )
    return df

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

if __name__ == "__main__":
    main()