# partition_by_speaker.py
import os
import pandas as pd
import re

def extract_speaker_id(audio_path: str) -> str:
    """
    Extract speaker ID from a LibriSpeech-style audio path.

    Works for paths like:
      /stek/corpora/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac
      /stek/corpora/LibriSpeech/train-clean-360/208/1700/208-1700-0000.flac
      /stek/corpora/LibriSpeech/train-other-500/6403/123456/6403-123456-0001.flac
    """
    parts = audio_path.strip().split("/")

    # Find the numeric folder after 'LibriSpeech/<subset>'
    if "LibriSpeech" in parts:
        idx = parts.index("LibriSpeech")
        for i in range(idx + 1, len(parts)):
            if parts[i].isdigit():
                return parts[i]

    # Fallback: regex pattern
    match = re.search(r"/(\d+)/\d+/\d+-\d+-\d+\.(flac|wav|mp3)$", audio_path)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract speaker id from path: {audio_path}")

def partition_by_speaker(csv_path: str, out_dir: str):
    """
    Partition dataset CSV into multiple CSV files — one per speaker.
    Output filenames: client_1.csv, client_2.csv, ...

    Args:
        csv_path: Path to input CSV (columns: dataset, set, audio_path, isspeech,
                  transcript, gender, emotion, age, accent, audio_len)
        out_dir: Directory to store output files.

    Returns:
        List of output client CSV file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Extract speaker IDs
    df["speaker_id"] = df["audio_path"].apply(extract_speaker_id)
    speaker_ids = sorted(df["speaker_id"].unique().tolist())

    print(f"Found {len(speaker_ids)} unique speakers.")

    client_csvs = []
    for i, spk_id in enumerate(speaker_ids, start=1):
        spk_df = df[df["speaker_id"] == spk_id].reset_index(drop=True)
        out_path = os.path.join(out_dir, f"client_{i}.csv")
        spk_df.to_csv(out_path, index=False)
        client_csvs.append(out_path)

    print(f"✅ Created {len(client_csvs)} client CSV files in '{out_dir}'")
    return client_csvs


if __name__ == "__main__":
    # Example usage
    input_csv = "./libri_csv/dev.csv"
    out_dir = "./fl_LS_dev_speaker"
    partition_by_speaker(input_csv, out_dir)
