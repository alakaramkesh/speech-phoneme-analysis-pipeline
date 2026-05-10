import os
import yaml
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from statsmodels.formula.api import mixedlm
import statsmodels.api as sm


# paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")
with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)
INPUT_CSV = os.path.join(BASE_DIR, params["normalization"]["output_csv"])
WHISPER_FILE = os.path.join(BASE_DIR, params["neural_analysis"]["whisper_file"])
XLSR_FILE = os.path.join(BASE_DIR, params["neural_analysis"]["xlsr_file"])
OUTPUT_DIR = os.path.join(BASE_DIR, "results/mixed_models")
VOWEL = "a"

# ICC
def compute_icc(model):
    speaker_variance = model.cov_re.iloc[0, 0]
    residual_variance = model.scale
    return speaker_variance / (speaker_variance + residual_variance)
# MARGINAL R2
def compute_marginal_r2(model, df):
    fixed_predictions = model.predict(df)
    fixed_variance = np.var(fixed_predictions)
    total_variance = fixed_variance +model.cov_re.iloc[0, 0] + model.scale
    return fixed_variance / total_variance
# ACOUSTIC MODEL
def run_acoustic_models():
    df = pd.read_csv(INPUT_CSV)
    df = df[df["label"] == VOWEL].copy()
    df = df.dropna(subset=["F1_norm", "speaker_id", "gender", "L1"])
    df["gender_binary"] = (df["gender"].str.lower() == "m").astype(int)
    df["L1_binary"] = (df["L1"] == "L2").astype(int)
    # ICC model
    null_model = sm.MixedLM.from_formula("F1_norm ~ 1", groups="speaker_id", data=df).fit(method="powell", reml=True)
    acoustic_icc = compute_icc(null_model)
    # main-effects model
    main_model = sm.MixedLM.from_formula("F1_norm ~ L1_binary + gender_binary", groups="speaker_id", data=df).fit(method="powell", reml=False)
    acoustic_r2 = compute_marginal_r2(main_model, df)
    try:
        full_model = sm.MixedLM.from_formula("F1_norm ~ L1_binary * gender_binary", groups="speaker_id", data=df).fit(method="powell", reml=False)
        # interaction model
        interaction_p = full_model.pvalues.get("L1_binary:gender_binary", np.nan)
    except Exception:
        interaction_p = np.nan
    return acoustic_icc, interaction_p, acoustic_r2
# NEURAL MODEL
def run_neural_model(npz_path, model_name):
    # load neural embeddings + metadata
    data = np.load(npz_path, allow_pickle=True)
    # keep only the selected vowel
    mask = data["labels"] == VOWEL
    embeddings = data["embeddings"][mask]
    speakers = data["speakers"][mask]
    groups = data["l1"][mask]
    genders = data["gender"][mask]
    # project embeddings to first 5 PCs
    pca = PCA(n_components=5)
    pcs = pca.fit_transform(embeddings)
    # create dataframe
    df = pd.DataFrame({
        "speaker_id": speakers,
        "L1_binary": (groups == "L2").astype(int),
        "gender": (genders == "m").astype(int)
    })
    # add PC columns
    for i in range(5):
        df[f"PC{i+1}"] = pcs[:, i]
    # store results for all PCs
    icc_values = []
    interaction_pvalues = []
    r2_values = []
    # fit one LME per PC dimension
    for i in range(5):
        pc_name = f"PC{i+1}"
        # null model
        null_model = sm.MixedLM.from_formula(f"{pc_name} ~ 1",groups="speaker_id",data=df).fit(method="powell", reml=True)
        icc_values.append(compute_icc(null_model))
        # full model
        full_model = sm.MixedLM.from_formula(f"{pc_name} ~ L1_binary * gender",groups="speaker_id",data=df).fit(method="powell", reml=False)
        # get interaction p-value
        interaction_pvalues.append(full_model.pvalues.get("L1_binary:gender",np.nan))
        # main-effects model
        main_model = sm.MixedLM.from_formula(f"{pc_name} ~ L1_binary + gender",groups="speaker_id",data=df).fit(method="powell", reml=False)
        r2_values.append(compute_marginal_r2(main_model, df))
    # average across the 5 PCs
    return {
        "model": model_name,
        "icc": np.mean(icc_values),
        "interaction_p": np.nanmean(interaction_pvalues),
        "marginal_r2": np.mean(r2_values)
    }
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # acoustic
    acoustic_icc, acoustic_interaction_p, acoustic_r2 = run_acoustic_models()
    # whisper
    whisper_results = run_neural_model(WHISPER_FILE,"whisper")
    # xlsr
    xlsr_results = run_neural_model(XLSR_FILE,"xlsr")
    # ICC results
    icc_df = pd.DataFrame({
        "representation": [
            "acoustic",
            "whisper",
            "xlsr"
        ],
        "ICC": [
            acoustic_icc,
            whisper_results["icc"],
            xlsr_results["icc"]
        ]
    })
    # interaction results
    interaction_df = pd.DataFrame({
        "representation": [
            "acoustic",
            "whisper",
            "xlsr"
        ],
        "interaction_p": [
            acoustic_interaction_p,
            whisper_results["interaction_p"],
            xlsr_results["interaction_p"]
        ]
    })
    # R2 results
    r2_df = pd.DataFrame({
        "representation": [
            "acoustic",
            "whisper",
            "xlsr"
        ],
        "marginal_r2": [
            acoustic_r2,
            whisper_results["marginal_r2"],
            xlsr_results["marginal_r2"]
        ]
    })
    icc_df.to_csv(os.path.join(OUTPUT_DIR, "icc_results.csv"),index=False)
    interaction_df.to_csv(os.path.join(OUTPUT_DIR, "interaction_results.csv"),index=False)
    r2_df.to_csv(os.path.join(OUTPUT_DIR, "r2_results.csv"),index=False)
    print(icc_df)
    print(interaction_df)
    print(r2_df)

if __name__ == "__main__":
    main()