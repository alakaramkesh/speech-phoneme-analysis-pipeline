import os
import pandas as pd
import parselmouth
from tqdm import tqdm
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")
with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)

# Load params
INPUT_CSV = os.path.join(BASE_DIR, params["acoustics"]["input_csv"])
OUTPUT_CSV = os.path.join(BASE_DIR, params["acoustics"]["output_csv"])

MAX_FORMANT_F = params["acoustics"]["max_formant_female"]
MAX_FORMANT_M = params["acoustics"]["max_formant_male"]

VOWELS = params["acoustics"]["vowels"]
FRICATIVES = params["acoustics"]["fricatives"]


def extract_formants(sound, time, gender):
    # Set max formant depending on gender
    max_formant = MAX_FORMANT_F if gender == "f" else MAX_FORMANT_M
    formant = sound.to_formant_burg(
        max_number_of_formants=5,
        maximum_formant=max_formant
    )

    # Extract F1 and F2 at time
    f1 = formant.get_value_at_time(1, time)
    f2 = formant.get_value_at_time(2, time)
    f3 = formant.get_value_at_time(3, time)
    return f1, f2 , f3

def extract_pitch(sound):
    # Extract pitch values (f0) from the audio
    pitch = sound.to_pitch()
    return pitch.selected_array['frequency']

def is_vowel(label):
    # Check if a phoneme is a vowel
    return label in VOWELS


def is_fricative(label):
    # Check if a phoneme is a fricative
    return label in FRICATIVES

def extract_scg(sound):
    # Compute spectral centre of gravity (SCG)
    spectrum = sound.to_spectrum()
    return spectrum.get_center_of_gravity()

def main():
    # Load phoneme table
    df = pd.read_csv(INPUT_CSV)
    rows = []
    missing_wavs = 0
    grouped = df.groupby("wav_path")

    for wav_path, group in tqdm(grouped, total=len(grouped)):

        if not os.path.exists(wav_path):
            missing_wavs += len(group)
            continue

        try:
            sound = parselmouth.Sound(wav_path) # Load audio
        except Exception:
            sound = None

        for _, row in group.iterrows():

            onset = row["onset"]
            offset = row["offset"]
            duration = row["duration"]
            midpoint = (onset + offset) / 2.0
            gender = row.get("gender", "m")
            label = row["label"]
            try:
                if sound is None:
                    raise Exception("Invalid audio")
                # Extract formants
                f1, f2, f3 = extract_formants(sound, midpoint, gender)
                # f3 (only vowels)
                if not is_vowel(label):
                    f3 = None
                # Take mean f0 over the whole signal
                pitch_values = extract_pitch(sound)
                f0 = pitch_values.mean() if len(pitch_values) > 0 else None
                # keep f0 if it is voiced
                if f0 is None or f0 <= 0:
                    f0 = None
                # Only for fricatives
                scg = extract_scg(sound) if is_fricative(label) else None
            except Exception:
                # Handle any extraction failure
                f1, f2, f3, f0, scg = None, None, None, None, None

            rows.append({
                "speaker_id": row["speaker_id"],
                "sentence_id": row["sentence_id"],
                "repetition": row["repetition"],
                "label": label,
                "onset": onset,
                "offset": offset,
                "duration": duration,
                "midpoint": midpoint,
                "F1": f1,
                "F2": f2,
                "F3": f3,
                "f0": f0,
                "SCG": scg,
                "gender": gender,
                "L1": row["L1"]
            })


    # Save output
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved to {OUTPUT_CSV}")
    # sanity checks(could be ignored)
    #missing_gender = df["gender"].isna().sum()
    #print(f"Missing gender: {missing_gender}")
    #print(f"Missing wav files: {missing_wavs}")


if __name__ == "__main__":
    main()