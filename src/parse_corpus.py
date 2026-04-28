import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "FRcorp_textgrids_only")# Path to the folder that contains one folder per speaker
META_PATH = os.path.join(BASE_DIR, "data", "raw", "metadata.csv")# Path where we save the final phoneme table
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "phoneme_table.csv")

def main(): 
    # We store each phoneme token as one row in this list
    rows = []
    # Loop over speaker folders
    for speaker_id in os.listdir(DATA_DIR):
        speaker_path = os.path.join(DATA_DIR, speaker_id)
        # Skip anything that is not a folder
        if not os.path.isdir(speaker_path):
            continue
        # Loop over files inside one speaker folder
        for file in os.listdir(speaker_path):
            # We only need TextGrid files here
            if not file.endswith(".TextGrid"):
                continue
            tg_path = os.path.join(speaker_path, file) # Full path to the TextGrid file
            wav_path = tg_path.replace(".TextGrid", ".wav") # The wav file should have the same name as the TextGrid file
            if not os.path.exists(wav_path):
                continue
            fname = file.replace(".TextGrid", "") # Remove the file extension and Split the filename into parts
            parts = fname.split("_")
            lang = parts[1] # parts[1] is the language group: fra or rus
            sentence_id = parts[-1] # The last part gives the sentence id
            L1 = "L1" if lang == "fra" else "L2"
            rep = parts[2] # repetition index e.g: list1
            rep_idx = int(rep.replace("list", ""))
            # Read the TextGrid as a normal text file
            with open(tg_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            # We will collect only intervals from the phones tier
            in_phones_tier = False
            current_interval = {}
            phoneme_list = []

            for line in lines:
                line = line.strip()
                if line.startswith('name = "phones"'): # Detect the start of the phones tier
                    in_phones_tier = True
                    continue
                if in_phones_tier and line.startswith("item ["): # in case there are more than two tiers skip them
                    break
                # Read interval information inside phones tier
                if in_phones_tier:
                    if line.startswith("xmin ="):
                        current_interval["onset"] = float(line.split("=")[1].strip())
                    elif line.startswith("xmax ="):
                        current_interval["offset"] = float(line.split("=")[1].strip())
                    elif line.startswith("text ="):
                        label = line.split("=", 1)[1].strip().strip('"')
                        # Skip empty and silence intervals
                        if label not in ["", "sil", "sp", "spn"] and "..." not in label and len(label) <= 3:
                            current_interval["label"] = label
                            phoneme_list.append(current_interval.copy())
                        current_interval = {} # just to be safe, we reset
            # creating a list of all phonemes
            for p in phoneme_list:
                rows.append({
                    "speaker_id": speaker_id,
                    "sentence_id": sentence_id,
                    "repetition": rep_idx,
                    "filename": fname,
                    "lang": lang,
                    "L1": L1,
                    "label": p["label"],
                    "onset": p["onset"],
                    "duration": p["offset"] - p["onset"],
                    "offset": p["offset"],
                    "wav_path": wav_path
                })

    df = pd.DataFrame(rows)
    meta = pd.read_csv(META_PATH, sep=";")
    meta = meta.rename(columns={
        "spk": "speaker_id",
        "Gender": "gender",
        "Age": "age"
    })
    df = df.merge(meta[["speaker_id", "gender", "age"]], on="speaker_id", how="left")
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()