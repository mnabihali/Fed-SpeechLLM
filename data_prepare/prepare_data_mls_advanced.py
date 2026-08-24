import os
import csv
import torchaudio

def prepare_mls_italian_full(data_folder, save_folder):
    """
    Prepares train_itmls.csv, dev_itmls.csv, test_itmls.csv
    for SpeechBrain training. Handles deep subfolder structures.
    """
    os.makedirs(save_folder, exist_ok=True)
    splits = {"train": "train_nlmls.csv", "dev": "dev_nlmls.csv", "test": "test_nlmls.csv"}

    for split, csv_name in splits.items():
        split_path = os.path.join(data_folder, split)
        transcript_file = os.path.join(split_path, "transcripts.txt")
        csv_out = os.path.join(save_folder, csv_name)

        if not os.path.exists(transcript_file):
            print(f"?? Transcript file not found for {split}: {transcript_file}")
            continue

        # ? Build a lookup dictionary for all .flac files in subfolders
        print(f"?? Indexing audio files for {split} ...")
        flac_map = {}
        for root, _, files in os.walk(os.path.join(split_path, "audio")):
            for fname in files:
                if fname.endswith(".flac"):
                    utt_id = os.path.splitext(fname)[0]
                    flac_map[utt_id] = os.path.join(root, fname)
        print(f"? Found {len(flac_map)} audio files in {split}")

        # ? Write CSV
        with open(csv_out, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "dataset", "set", "audio_path", "isspeech",
                "transcript", "gender", "emotion", "age", "accent", "audio_len"
            ])
            writer.writeheader()

            count = 0
            with open(transcript_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    utt_id, text = parts

                    audio_path = flac_map.get(utt_id)
                    if not audio_path:
                        continue

                    try:
                        info = torchaudio.info(audio_path)
                        duration = info.num_frames / info.sample_rate
                    except Exception:
                        duration = 0.0

                    writer.writerow({
                        "dataset": "mls",
                        "set": split,
                        "audio_path": audio_path,
                        "isspeech": "TRUE",
                        "transcript": text.strip(),
                        "gender": "__unknown__",
                        "emotion": "__unknown__",
                        "age": "__unknown__",
                        "accent": "__unknown__",
                        "audio_len": round(duration, 2)
                    })

                    count += 1
                    if count % 1000 == 0:
                        print(f"?? Processed {count} utterances...")

        print(f"? Created {csv_out} ({count} entries)")


prepare_mls_italian_full("/stek/corpora/mls_dutch", "/stek/mohamed/SpeechLLM/csv_mlsdutch")
