import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.spatial.distance import cosine, pdist
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="The random effects covariance matrix is singular")
warnings.filterwarnings("ignore", message="Random effects covariance is singular")
warnings.filterwarnings("ignore", message="The MLE may be on the boundary")
warnings.filterwarnings("ignore", message="The Hessian matrix at the estimated parameter values is not positive definite")
from scipy.optimize import brentq
CHI2_THRESHOLD = 3.841
BISECT_TOL = 1e-6
# paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")
with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)
INPUT_CSV = os.path.join(BASE_DIR, params["normalization"]["output_csv"])
WHISPER_FILE = os.path.join(BASE_DIR, params["neural_analysis"]["whisper_file"])
XLSR_FILE = os.path.join(BASE_DIR, params["neural_analysis"]["xlsr_file"])
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "results/figures")
VOWELS = params["acoustics"]["vowels"]
BOOTSTRAP_ITERATIONS = params["section8"]["bootstrap_iterations"]
RANDOM_SEED = params["section8"]["random_seed"]
# simple ROPE for Lobanov-normalized formants
ROPE = (params["section8"]["acoustic_rope"]["low"],params["section8"]["acoustic_rope"]["high"])
# PROFILE LOG-LIKELIHOOD PENALTY
def profile_penalty(theta, term, df, formant, loglik_max):
    df_copy = df.copy()
    if term == "L1_binary":
        df_copy["offset_response"] = df_copy[formant] - theta * df_copy["L1_binary"]
        formula = "offset_response ~ gender_binary"
    else:
        interaction = df_copy["L1_binary"] * df_copy["gender_binary"]
        df_copy["offset_response"] = df_copy[formant] - theta * interaction
        formula = "offset_response ~ L1_binary + gender_binary"
    try:
        model = sm.MixedLM.from_formula(formula, groups="speaker_id", data=df_copy).fit(method="powell", reml=False)
        return 2 * (loglik_max - model.llf)
    except Exception:
        return 1e6
# PROFILE LIKELIHOOD CI
def profile_likelihood_ci(model, term, df, formant):
    estimate = model.params[term]
    se = model.bse[term]
    loglik_max = model.llf
    lower_guess = estimate - 3 * se
    upper_guess = estimate + 3 * se
    def lower_function(theta): return profile_penalty(theta, term, df, formant, loglik_max) - CHI2_THRESHOLD
    def upper_function(theta): return profile_penalty(theta, term, df, formant, loglik_max) - CHI2_THRESHOLD
    try:
        ci_low = brentq(lower_function, lower_guess, estimate, xtol=BISECT_TOL)
        ci_high = brentq(upper_function, estimate, upper_guess, xtol=BISECT_TOL)
        return ci_low, ci_high
    except Exception:
        fallback = model.conf_int().loc[term]
        return fallback[0], fallback[1]
# FIT ONE MIXED MODEL
def fit_model(df, formant):
    try: return sm.MixedLM.from_formula(f"{formant} ~ L1_binary * gender_binary", groups="speaker_id", data=df).fit(method="powell", reml=False)
    except Exception:
        print(f"Falling back to simpler model for {formant}")
        return sm.MixedLM.from_formula(f"{formant} ~ L1_binary + gender_binary", groups="speaker_id", data=df).fit(method="powell", reml=False)
# EXTRACT CIs + EFFECTS
def extract_contrasts(model, df, phoneme, formant, n_speakers, n_tokens):
    rows = []
    # statsmodels provides Wald confidence intervals here.
    #ci = model.conf_int()
    params = model.params
    pvalues = model.pvalues
    for term in ["L1_binary", "L1_binary:gender_binary"]:
        if term not in params.index:
            continue
        estimate = params[term]
        #low, high = ci.loc[term]
        # using profile likelihood CI
        low, high = profile_likelihood_ci(model, term, df, formant)
        p_value = pvalues[term]
        # This tells whether the effect is practically meaningful.
        if low >= ROPE[0] and high <= ROPE[1]:
            rope_class = "Equivalent"
        elif high < ROPE[0] or low > ROPE[1]:
            rope_class = "Non-equivalent"
        else:
            rope_class = "Indeterminate"
        rows.append({
            "phoneme": phoneme,
            "formant": formant,
            "term": term,
            "estimate": estimate,
            "ci_low": low,
            "ci_high": high,
            "p_value": p_value,
            "rope_classification": rope_class,
            "n_speakers": n_speakers,
            "n_tokens": n_tokens                    
        })
    return rows
# RUN ACOUSTIC CI ANALYSIS
def run_acoustic_ci():
    df = pd.read_csv(INPUT_CSV)
    # keep only vowels
    df = df[df["label"].isin(VOWELS)].copy()
    # binary variables
    df["gender_binary"] = (df["gender"].astype(str).str.lower().str.startswith("m")).astype(int)
    df["L1_binary"] = (df["L1"].astype(str).str.upper() == "L2").astype(int)
    rows = []
    for phoneme in sorted(df["label"].unique()):
        vowel_df = df[df["label"] == phoneme].copy()
        for formant in ["F1_norm", "F2_norm"]:
            vowel_df_clean = vowel_df.dropna(
                subset=[
                    formant,
                    "speaker_id",
                    "gender_binary",
                    "L1_binary"
                ]
            )
            if len(vowel_df_clean) < 10 or vowel_df_clean["speaker_id"].nunique() < 3:
                print(f"Skipping {phoneme} {formant}")
                continue
            try:
                model = fit_model(vowel_df_clean, formant)
                rows.extend(extract_contrasts(
                    model=model,
                    df=vowel_df_clean,
                    phoneme=phoneme,
                    formant=formant,
                    n_speakers=vowel_df_clean["speaker_id"].nunique(),
                    n_tokens=len(vowel_df_clean)
                ))
            except Exception as e:
                print(f"Error for {phoneme} {formant}: {e}")
    return pd.DataFrame(rows)
# FOREST PLOT
def make_forest_plot(results_df, formant, term):
    plot_df = results_df[
        (results_df["formant"] == formant) &
        (results_df["term"] == term)
    ].copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("estimate")
    y = np.arange(len(plot_df))
    plt.figure(figsize=(8, max(4, len(plot_df) * 0.4)))
    # CI bars
    plt.errorbar(
        plot_df["estimate"],
        y,
        xerr=[
            plot_df["estimate"] - plot_df["ci_low"],
            plot_df["ci_high"] - plot_df["estimate"]
        ],
        fmt="o",
        capsize=4
    )
    # no-effect line
    plt.axvline(0, linestyle="--")
    # ROPE region
    plt.axvspan(
        ROPE[0],
        ROPE[1],
        alpha=0.15
    )
    plt.yticks(y, plot_df["phoneme"])
    plt.xlabel("Effect size")
    plt.ylabel("Phoneme")
    plt.title(f"{formant} - {term}")
    plt.tight_layout()
    safe_term = term.replace(":", "_")
    out_path = os.path.join(FIGURES_DIR,f"forest_{formant}_{safe_term}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# SAFE COSINE DISTANCE
def cosine_distance(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return np.nan
    return cosine(a, b)

# SPEAKER-LEVEL MEANS
def compute_speaker_means(embeddings, labels, speakers, groups, phoneme):
    mask = labels == phoneme
    X = embeddings[mask]
    spk = speakers[mask]
    grp = groups[mask]
    rows = []
    for speaker in np.unique(spk):
        speaker_mask = spk == speaker
        rows.append({
            "speaker_id": speaker,
            "L1": grp[speaker_mask][0],
            "vector": X[speaker_mask].mean(axis=0)
        })
    return pd.DataFrame(rows)

# MEAN INTRA-SPEAKER DISTANCE
def compute_noise_floor(embeddings, labels, speakers, phoneme):
    mask = labels == phoneme
    X = embeddings[mask]
    spk = speakers[mask]
    distances = []
    for speaker in np.unique(spk):
        speaker_X = X[spk == speaker]
        if len(speaker_X) < 2:
            continue
        distances.extend(
            pdist(speaker_X, metric="cosine")
        )
    if len(distances) == 0:
        return np.nan
    return np.nanmean(distances)

# BOOTSTRAP CI
def bootstrap_neural_ci(speaker_df):
    rng = np.random.default_rng(RANDOM_SEED)
    l1_df = speaker_df[speaker_df["L1"] == "L1"].reset_index(drop=True)
    l2_df = speaker_df[speaker_df["L1"] == "L2"].reset_index(drop=True)
    if len(l1_df) < 2 or len(l2_df) < 2:
        return np.nan, np.nan, np.nan
    observed_l1 = np.vstack(l1_df["vector"]).mean(axis=0)
    observed_l2 = np.vstack(l2_df["vector"]).mean(axis=0)
    observed_distance = cosine_distance(
        observed_l1,
        observed_l2
    )
    bootstrap_values = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        l1_sample = l1_df.iloc[rng.integers(0, len(l1_df), len(l1_df))]
        l2_sample = l2_df.iloc[rng.integers(0, len(l2_df), len(l2_df))]
        l1_centroid = np.vstack(l1_sample["vector"]).mean(axis=0)
        l2_centroid = np.vstack(l2_sample["vector"]).mean(axis=0)
        bootstrap_values.append(cosine_distance(l1_centroid, l2_centroid))
    ci_low, ci_high = np.nanpercentile(bootstrap_values,[2.5, 97.5])
    return observed_distance, ci_low, ci_high
# NEURAL ROPE CLASSIFICATION
def classify_neural_rope(ci_low, ci_high, rope_high):
    if np.isnan(ci_low) or np.isnan(ci_high) or np.isnan(rope_high):
        return "Missing"
    if ci_high <= rope_high:
        return "Equivalent"
    if ci_low > rope_high:
        return "Non-equivalent"
    return "Indeterminate"
# RUN NEURAL CI ANALYSIS
def run_neural_ci(npz_path, model_name):
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    speakers = data["speakers"]
    groups = data["l1"]
    rows = []
    for phoneme in VOWELS:
        if phoneme not in labels:
            continue
        speaker_df = compute_speaker_means(embeddings,labels,speakers,groups,phoneme)
        if speaker_df.empty:
            continue
        estimate, ci_low, ci_high = bootstrap_neural_ci(speaker_df)
        rope_high = compute_noise_floor(embeddings,labels,speakers,phoneme)
        rope_class = classify_neural_rope(ci_low,ci_high,rope_high)
        rows.append({
            "phoneme": phoneme,
            "representation": model_name,
            "estimate": estimate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "rope_low": 0.0,
            "rope_high": rope_high,
            "rope_classification": rope_class,
            "n_speakers": speaker_df["speaker_id"].nunique()
        })
    return pd.DataFrame(rows)
# NEURAL FOREST PLOT
def make_neural_forest_plot(results_df, model_name):
    plot_df = results_df[
        results_df["representation"] == model_name
    ].copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("estimate")
    y = np.arange(len(plot_df))
    plt.figure(figsize=(8, max(4, len(plot_df) * 0.4)))
    plt.errorbar(
        plot_df["estimate"],
        y,
        xerr=[
            plot_df["estimate"] - plot_df["ci_low"],
            plot_df["ci_high"] - plot_df["estimate"]
        ],
        fmt="o",
        capsize=4
    )
    plt.axvline(0, linestyle="--")
    plt.axvline(plot_df["rope_high"].mean(),linestyle=":")
    plt.yticks(y, plot_df["phoneme"])
    plt.xlabel("Cosine distance")
    plt.ylabel("Phoneme")
    plt.title(f"{model_name} neural contrasts")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR,f"forest_neural_{model_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    acoustic_results_df = run_acoustic_ci()
    if acoustic_results_df.empty: print("No valid acoustic models were fitted.")
    acoustic_csv = os.path.join(OUTPUT_DIR, "acoustic_ci_results.csv")
    acoustic_results_df.to_csv(acoustic_csv, index=False)
    print(acoustic_results_df.head())
    for formant in ["F1_norm", "F2_norm"]:
        for term in ["L1_binary", "L1_binary:gender_binary"]:
            make_forest_plot(acoustic_results_df, formant, term)
    whisper_results_df = run_neural_ci(WHISPER_FILE, "whisper")
    xlsr_results_df = run_neural_ci(XLSR_FILE, "xlsr")
    neural_results_df = pd.concat([whisper_results_df, xlsr_results_df], ignore_index=True)
    neural_csv = os.path.join(OUTPUT_DIR, "neural_ci_results.csv")
    neural_results_df.to_csv(neural_csv, index=False)
    print(neural_results_df.head())
    make_neural_forest_plot(neural_results_df, "whisper")
    make_neural_forest_plot(neural_results_df, "xlsr")
    combined_summary = pd.concat([acoustic_results_df.assign(representation="acoustic"), neural_results_df], ignore_index=True, sort=False)
    combined_summary.to_csv(os.path.join(OUTPUT_DIR, "section8_rope_summary.csv"), index=False)

if __name__ == "__main__":
    main()