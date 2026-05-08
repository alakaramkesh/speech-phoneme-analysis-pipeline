import os
import pandas as pd
import numpy as np
import yaml

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")
with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)
INPUT_CSV = os.path.join(BASE_DIR, params["descriptive_stats"]["input_csv"])
OUTPUT_CSV = os.path.join(BASE_DIR, params["descriptive_stats"]["output_csv"])
# helper functions
def compute_iqr(series):
    # Compute interquartile range
    q75 = series.quantile(0.75)
    q25 = series.quantile(0.25)
    return q75 - q25

def compute_cv(series):
    # Compute coefficient of variation
    mean_value = series.mean()
    std_value = series.std()
    # Avoid division by zero
    if mean_value == 0 or np.isnan(mean_value):
        return np.nan
    return std_value / mean_value

def main():
    # Load normalized acoustic features
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded rows: {len(df)}")
    results = []
    # Group by phoneme, L1 status, and gender
    grouped = df.groupby(["label", "L1", "gender"])
    for (label, l1_status, gender), group_df in grouped:
        # Compute total variance
        total_var = group_df["F1_norm"].var()
        # Compute inter-speaker variance
        speaker_means = group_df.groupby("speaker_id")["F1_norm"].mean()
        inter_var = speaker_means.var()
        # Compute intra-speaker variance
        intra_vars = group_df.groupby("speaker_id")["F1_norm"].var()
        intra_var = intra_vars.mean()
        # Compute residual variance
        residual_var = total_var - inter_var - intra_var

        row = {
            "label": label,
            "L1": l1_status,
            "gender": gender,
            "n_tokens": len(group_df),
            # F1 statistics
            "F1_mean": group_df["F1_norm"].mean(),
            "F1_median": group_df["F1_norm"].median(),
            "F1_std": group_df["F1_norm"].std(),
            "F1_iqr": compute_iqr(group_df["F1_norm"]),
            "F1_cv": compute_cv(group_df["F1_norm"]),
            # F2 statistics
            "F2_mean": group_df["F2_norm"].mean(),
            "F2_median": group_df["F2_norm"].median(),
            "F2_std": group_df["F2_norm"].std(),
            "F2_iqr": compute_iqr(group_df["F2_norm"]),
            "F2_cv": compute_cv(group_df["F2_norm"]),
             # Variance decomposition
            "F1_total_variance": total_var,
            "F1_inter_speaker_variance": inter_var,
            "F1_intra_speaker_variance": intra_var,
            "F1_residual_variance": residual_var,
        }
        results.append(row)
    results_df = pd.DataFrame(results)
    results_df = results_df.round(3)  # Round values for readability
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved descriptive statistics to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()