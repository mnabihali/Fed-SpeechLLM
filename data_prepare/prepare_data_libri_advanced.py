import os
import csv
import soundfile as sf

# Path to your LibriSpeech dataset
LIBRISPEECH_ROOT = "/stek/corpora/LibriSpeech"

# Path to the SPEAKERS.TXT file
SPEAKERS_FILE = os.path.join(LIBRISPEECH_ROOT, "SPEAKERS.TXT")

# Output CSV files
OUTPUT_CSVS = {
    "train": "train_100_360.csv",
    "dev": "dev.csv",
    "test": "test.csv"
}

# Subsets to process
SUBSETS = {
    "train-clean-100": "train",
    "train-clean-360": "train",
    #"train-other-500": "train",
    "dev-clean": "dev",
    "test-clean": "test"
}

# --- Load speaker genders from SPEAKERS.TXT ---
speaker_genders = {}
with open(SPEAKERS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith(";") or line == "":
            continue  # skip header/comments
        parts = [p.strip() for p in line.split("|")]
        speaker_id = parts[0]
        sex = parts[1]
        if sex.upper() == "F":
            speaker_genders[speaker_id] = "female"
        elif sex.upper() == "M":
            speaker_genders[speaker_id] = "male"
        else:
            speaker_genders[speaker_id] = "__unknown__"

# Function to gather all audio files in a folder
def gather_audio_files(root_dir, extensions=(".flac",)):
    audio_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(extensions):
                audio_files.append(os.path.join(dirpath, f))  # full path
    return audio_files

# Function to read transcript from chapter transcript file
def get_transcript(audio_file):
    folder = os.path.dirname(audio_file)
    base_name = os.path.basename(audio_file).replace(".flac", "")
    trans_file = os.path.join(folder, f"{base_name.rsplit('-', 1)[0]}.trans.txt")
    
    if os.path.exists(trans_file):
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(base_name):
                    return line.strip().split(" ", 1)[1]
    return "__unknown__"

# Prepare a CSV writer for each set
writers = {}
files = {}
for set_name, csv_path in OUTPUT_CSVS.items():
    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["dataset", "set", "audio_path", "isspeech", "transcript",
                     "gender", "emotion", "age", "accent", "audio_len"])
    writers[set_name] = writer
    files[set_name] = f

# Process each subset
for subset_folder, set_type in SUBSETS.items():
    subset_path = os.path.join(LIBRISPEECH_ROOT, subset_folder)
    if not os.path.exists(subset_path):
        print(f"Warning: {subset_path} does not exist, skipping.")
        continue
    
    audio_files = gather_audio_files(subset_path)
    
    for audio_file in audio_files:
        try:
            info = sf.info(audio_file)
            audio_len = round(info.frames / info.samplerate, 2)
        except:
            audio_len = "__unknown__"
        
        transcript = get_transcript(audio_file)
        
        # Extract speaker ID from path: LibriSpeech/.../speaker_id/chapter_id/...
        parts = audio_file.split(os.sep)
        speaker_id = None
        for p in parts:
            if p.isdigit() and p in speaker_genders:
                speaker_id = p
                break
        gender = speaker_genders.get(speaker_id, "__unknown__")
        
        writers[set_type].writerow([
            "librispeech",
            set_type,
            audio_file,       # full path to audio
            True,
            transcript,
            gender,           # gender from SPEAKERS.TXT
            "__unknown__",    # emotion
            "__unknown__",    # age
            "__unknown__",    # accent
            audio_len
        ])

# Close all files
for f in files.values():
    f.close()

print("CSV files created: train.csv, dev.csv, test.csv with updated gender")
