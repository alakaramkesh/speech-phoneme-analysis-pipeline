import os
import pandas as pd
import numpy as np
import yaml
#paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")
with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)
INPUT_CSV = os.path.join(BASE_DIR, params["normalization"]["input_csv"])
OUTPUT_CSV = os.path.join(BASE_DIR, params["normalization"]["output_csv"])
VOWELS = params["acoustics"]["vowels"]

def lobanov_normalize(df, formant_col): # applying lobanov norm on one column
    normalized_values = []
    for speaker_id, speaker_df in df.groupby("speaker_id"):
        mean_value = speaker_df[formant_col].mean()
        std_value = speaker_df[formant_col].std()
        # avoid division by 0
        if std_value == 0 or np.isnan(std_value):
            normalized = speaker_df[formant_col] - mean_value
        else:
            normalized = (speaker_df[formant_col] - mean_value) / std_value
        normalized_values.append(normalized) # Store normalized values for this speaker

    return pd.concat(normalized_values).sort_index() # Merge all normalized values while preserving row order


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Initial rows: {len(df)}")
    vowel_df = df[df["label"].isin(VOWELS)].copy()
    print(f"Vowel rows before dropping missing values: {len(vowel_df)}")
    vowel_df = vowel_df.dropna(subset=["F1", "F2"]) # remove rows with missing F1 or F2 values
    print(f"Vowel rows after dropping missing values: {len(vowel_df)}")
    # apply the lobanov norm and save results in new columns
    vowel_df["F1_norm"] = lobanov_normalize(vowel_df, "F1")
    vowel_df["F2_norm"] = lobanov_normalize(vowel_df, "F2")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    vowel_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved normalized acoustic features to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()