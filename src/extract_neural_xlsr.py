import os
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import yaml
import librosa


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")
INPUT_CSV = os.path.join(BASE_DIR, "data/processed/phoneme_table.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/features")

with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)

# XLS-R settings
MODEL_NAME = params["xlsr"]["model_name"]
LAYERS = params["xlsr"]["layers"]


def load_xlsr_model(model_name):
    # Load XLS-R feature extractor and model
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    # Use GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device


def get_hidden_states_for_wav(wav_path, processor, model, device):
    # Load audio file
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:  # XLS-R expects mono audio
        audio = audio.mean(axis=1)
    if sr != 16000:  # XLS-R expects 16 kHz audio
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    # Prepare audio input for XLS-R
    inputs = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )
    input_values = inputs.input_values.to(device)
    # Run model and get hidden states from all layers
    with torch.no_grad():
        outputs = model(
            input_values,
            output_hidden_states=True
        )
    # hidden_states is a list: one tensor per transformer layer
    hidden_states = outputs.hidden_states
    # Audio duration in seconds
    audio_duration = len(audio) / sr
    return hidden_states, audio_duration


def extract_token_embedding(hidden, audio_duration, onset, offset):
    num_frames = hidden.shape[0]  # Number of XLS-R time steps
    frame_duration = audio_duration / num_frames  # Approximate duration of one frame
    # Convert phoneme time boundaries to frame indices
    start_frame = int(onset / frame_duration)
    end_frame = int(offset / frame_duration)
    # Keep indices inside valid range (sanity check)
    start_frame = max(0, min(start_frame, num_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, num_frames))
    # Select hidden states overlapping with the phoneme
    frames = hidden[start_frame:end_frame]
    # Average-pool across time
    embedding = frames.mean(axis=0)
    return embedding


def main():
    df = pd.read_csv(INPUT_CSV)  # Load phoneme table
    processor, model, device = load_xlsr_model(MODEL_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output folder
    # Store embeddings separately for each layer
    layer_embeddings = {layer_idx: [] for layer_idx in LAYERS}
    layer_token_ids = {layer_idx: [] for layer_idx in LAYERS}
    layer_labels = {layer_idx: [] for layer_idx in LAYERS}
    layer_l1 = {layer_idx: [] for layer_idx in LAYERS}
    layer_gender = {layer_idx: [] for layer_idx in LAYERS}
    layer_speakers = {layer_idx: [] for layer_idx in LAYERS}
    # Process one wav file at a time for efficiency
    grouped = df.groupby("wav_path")
    for wav_path, group in tqdm(grouped, total=len(grouped)):
        if not os.path.exists(wav_path):
            print(f"Missing wav file: {wav_path}")
            continue
        try:
            hidden_states, audio_duration = get_hidden_states_for_wav(
                wav_path,
                processor,
                model,
                device
            )
            # Convert selected layers once per wav
            selected_hidden = {}
            for layer_idx in LAYERS:
                selected_hidden[layer_idx] = hidden_states[layer_idx][0].cpu().numpy()
            # Extract one embedding per phoneme token, for each selected layer
            for idx, row in group.iterrows():
                for layer_idx in LAYERS:
                    emb = extract_token_embedding(
                        selected_hidden[layer_idx],
                        audio_duration,
                        row["onset"],
                        row["offset"]
                    )
                    layer_embeddings[layer_idx].append(emb)
                    layer_token_ids[layer_idx].append(idx)
                    layer_labels[layer_idx].append(row["label"])
                    layer_l1[layer_idx].append(row["L1"])
                    layer_gender[layer_idx].append(row["gender"])
                    layer_speakers[layer_idx].append(row["speaker_id"])
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
    # Save one NPZ file per layer
    for layer_idx in LAYERS:
        embeddings = np.stack(layer_embeddings[layer_idx])
        token_ids = np.array(layer_token_ids[layer_idx])
        output_path = os.path.join(
            OUTPUT_DIR,
            f"features_xlsr_layer{layer_idx}.npz"
        )
        np.savez(
            output_path,
            embeddings=embeddings,
            token_ids=token_ids,
            labels=np.array(layer_labels[layer_idx]),
            l1=np.array(layer_l1[layer_idx]),
            gender=np.array(layer_gender[layer_idx]),
            speakers=np.array(layer_speakers[layer_idx]),
            layer=np.array([layer_idx]),
            model=np.array([MODEL_NAME])
        )
        print(f"Saved: {output_path}")
        print(f"Embeddings shape: {embeddings.shape}")

if __name__ == "__main__":
    main()