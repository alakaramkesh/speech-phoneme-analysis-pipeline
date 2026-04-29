import os
import pandas as pd
import parselmouth
from tqdm import tqdm

# Paths 
INPUT_CSV = "data/processed/phoneme_table.csv"
OUTPUT_CSV = "data/features/features_acoustic.csv"


def extract_formants(sound, time, gender):
    # Set max formant depending on gender
    max_formant = 5000 if gender == "f" else 4500

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
    vowels = ["a", "e", "i", "o", "u", "y", "ɑ", "ɛ", "ɔ", "œ", "ø", "ɯ", "ɪ", "ʊ"]
    return label in vowels


def is_fricative(label):
    # Check if a phoneme is a fricative
    return label in ["s", "ʃ", "z", "ʒ", "f", "v"]

def extract_scg(sound):
    # Compute spectral centre of gravity (SCG)
    spectrum = sound.to_spectrum()
    return spectrum.get_center_of_gravity()

def main():
    # Load phoneme table
    df = pd.read_csv(INPUT_CSV)
    rows = []
    missing_wavs = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        wav_path = row["wav_path"]
        # Skip if wav file does not exist
        if not os.path.exists(wav_path):
            missing_wavs += 1
            continue

        try:
            # Load audio
            sound = parselmouth.Sound(wav_path)
            onset = row["onset"]
            offset = row["offset"]
            duration = row["duration"]
            midpoint = (onset + offset) / 2.0
            gender = row.get("gender", "m")
            # Extract formants
            f1, f2, f3 = extract_formants(sound, midpoint, gender)
            # f3 (only vowels)
            label = row["label"]
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