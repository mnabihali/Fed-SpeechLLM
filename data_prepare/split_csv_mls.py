# partition_by_speaker.py
import os
import pandas as pd
import re

def extract_speaker_id(audio_path: str) -> str:
    # Ensure audio_path is a proper string
    if not isinstance(audio_path, str):
        raise ValueError(f"Invalid audio_path (not a string): {audio_path}")

    audio_path = audio_path.strip()

    parts = audio_path.split("/")

    # --- Case 1: MLS multilingual datasets ---
    if "audio" in parts:
        idx = parts.index("audio")
        if idx + 1 < len(parts) and parts[idx + 1].isdigit():
            return parts[idx + 1]

    # --- Case 2: Standard LibriSpeech ---
    if "LibriSpeech" in parts:
        idx = parts.index("LibriSpeech")
        for i in range(idx + 1, len(parts)):
            if parts[i].isdigit():
                return parts[i]

    # --- Fallback regex ---
    match = re.search(r"/(\d+)/\d+/[^/]+\.(flac|wav|mp3)$", audio_path)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract speaker id from path: {audio_path}")


def safe_extract(path):
    """Catch errors and print the broken path."""
    try:
        return extract_speaker_id(path)
    except Exception as e:
        print("\n? ERROR while processing audio_path:")
        print(f"   Raw value: {repr(path)}")
        print(f"   Error: {e}\n")
        raise e


def partition_by_speaker(csv_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    print(f"?? Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    print("?? Cleaning audio_path column...")

    # Convert all paths to strongly typed str
    df["audio_path"] = df["audio_path"].astype(str).str.strip()

    # Remove invalid rows
    df = df[df["audio_path"].str.len() > 0]
    df = df[df["audio_path"] != "nan"]
    df = df[~df["audio_path"].str.contains(r"^\s*$", regex=True)]

    print("?? Extracting speaker IDs (with debug)...")
    df["speaker_id"] = df["audio_path"].apply(safe_extract)

    speaker_ids = sorted(df["speaker_id"].unique().tolist())
    print(f"?? Found {len(speaker_ids)} unique speakers.")

    client_csvs = []
    for i, spk_id in enumerate(speaker_ids, start=1):
        spk_df = df[df["speaker_id"] == spk_id].reset_index(drop=True)
        out_path = os.path.join(out_dir, f"client_{i}.csv")
        spk_df.to_csv(out_path, index=False)
        client_csvs.append(out_path)

    print(f"? Created {len(client_csvs)} client CSV files in '{out_dir}'")
    return client_csvs


if __name__ == "__main__":
    input_csv = "/stek/mohamed/SpeechLLM/csv_mlsitalian/dev_itmls.csv"
    out_dir = "/stek/mohamed/FL-SLAM/fl_MLS_dev_speaker/"
    partition_by_speaker(input_csv, out_dir)

