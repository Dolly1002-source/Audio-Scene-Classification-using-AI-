import os
import shutil
import random
import zipfile
from collections import defaultdict

# =========================================================
# ALL ZIP PATHS
# =========================================================

AUDIO_ZIPS = [
    r"C:\Users\nimma\Downloads\m.s PROJECT\TUT-acoustic-scenes-2017-development.audio.1 (2).zip",
    r"C:\Users\nimma\Downloads\m.s PROJECT\TUT-acoustic-scenes-2017-development.audio.2 (2).zip",
    r"C:\Users\nimma\Downloads\m.s PROJECT\TUT-acoustic-scenes-2017-development.audio.3 (2).zip",
    r"C:\Users\nimma\Downloads\m.s PROJECT\TUT-acoustic-scenes-2017-development.audio.4 (2).zip",
    r"C:\Users\nimma\Downloads\m.s PROJECT\TUT-acoustic-scenes-2017-development.audio.5 (2).zip",
    r"C:\Users\nimma\Downloads\m.s PROJECT\TUT-acoustic-scenes-2017-development.audio.6 (1).zip",
    r"C:\Users\nimma\Downloads\m.s PROJECT\TUT-acoustic-scenes-2017-development.audio.7 (1).zip",
    r"C:\Users\nimma\Downloads\m.s PROJECT\TUT-acoustic-scenes-2017-development.audio.8 (1).zip",
    r"C:\Users\nimma\Downloads\m.s PROJECT\TUT-acoustic-scenes-2017-development.audio.9 (1).zip",
    r"C:\Users\nimma\Downloads\m.s PROJECT\TUT-acoustic-scenes-2017-development.audio.10 (2).zip",
]

META_ZIP = r"C:\Users\nimma\Downloads\m.s PROJECT\TUT-acoustic-scenes-2017-development.meta (1).zip"
ERROR_ZIP = r"C:\Users\nimma\Downloads\m.s PROJECT\TUT-acoustic-scenes-2017-development.error.zip"

# =========================================================
# OUTPUT / EXTRACT FOLDERS
# =========================================================

AUDIO_EXTRACT_DIR = r"C:\Users\nimma\Downloads\m.s PROJECT\TUT_audio_extracted"
META_EXTRACT_DIR = r"C:\Users\nimma\Downloads\m.s PROJECT\TUT_meta_extracted"
ERROR_EXTRACT_DIR = r"C:\Users\nimma\Downloads\m.s PROJECT\TUT_error_extracted"
OUTPUT_DIR = r"C:\Users\nimma\Downloads\m.s PROJECT\dataset_organized"

# =========================================================
# SPLIT RATIOS
# =========================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

random.seed(42)

# =========================================================
# HELPERS
# =========================================================

def extract_zip(zip_path, extract_to):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found:\n{zip_path}")

    os.makedirs(extract_to, exist_ok=True)

    print(f"Extracting:\n{zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to:\n{extract_to}\n")


def find_file(root_dir, filename):
    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def find_audio_dirs(root_dir):
    audio_dirs = []
    for root, dirs, files in os.walk(root_dir):
        if os.path.basename(root).lower() == "audio":
            audio_dirs.append(root)
    return audio_dirs


def build_audio_index(audio_dirs):
    """
    Build a recursive index:
      wav filename -> full path
    """
    index = {}
    duplicates = defaultdict(list)
    total_wavs_seen = 0

    for audio_dir in audio_dirs:
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                if file.lower().endswith(".wav"):
                    total_wavs_seen += 1
                    full_path = os.path.join(root, file)
                    if file in index:
                        duplicates[file].append(full_path)
                    else:
                        index[file] = full_path

    return index, duplicates, total_wavs_seen


# =========================================================
# STEP 1: EXTRACT ALL AUDIO ZIPs
# =========================================================

print("Extracting all audio ZIP parts...\n")
for zip_path in AUDIO_ZIPS:
    extract_zip(zip_path, AUDIO_EXTRACT_DIR)

print("Extracting meta ZIP...\n")
extract_zip(META_ZIP, META_EXTRACT_DIR)

print("Extracting error ZIP...\n")
extract_zip(ERROR_ZIP, ERROR_EXTRACT_DIR)

# =========================================================
# STEP 2: FIND meta.txt, error.txt, and all audio folders
# =========================================================

audio_dirs = find_audio_dirs(AUDIO_EXTRACT_DIR)
meta_file = find_file(META_EXTRACT_DIR, "meta.txt")
error_file = find_file(ERROR_EXTRACT_DIR, "error.txt")

if not audio_dirs:
    raise FileNotFoundError("Could not find any 'audio' folders after extracting the audio ZIP files.")

if meta_file is None:
    raise FileNotFoundError("meta.txt not found after extracting the meta ZIP.")

if error_file is None:
    raise FileNotFoundError("error.txt not found after extracting the error ZIP.")

print("Audio folders found:")
for d in audio_dirs:
    print(" ", d)

print(f"\nmeta.txt found at:\n{meta_file}")
print(f"error.txt found at:\n{error_file}\n")

# =========================================================
# STEP 3: INDEX ALL WAV FILES
# =========================================================

print("Indexing audio files recursively...")
audio_index, duplicates, total_wavs_seen = build_audio_index(audio_dirs)
print(f"Total wav files seen: {total_wavs_seen}")
print(f"Unique wav filenames indexed: {len(audio_index)}")

if duplicates:
    print(f"Warning: {len(duplicates)} duplicate filenames found.")
    print("First few duplicates:")
    for k, v in list(duplicates.items())[:5]:
        print(f"  {k}")
        for path in v[:3]:
            print(f"    {path}")
print()

# =========================================================
# STEP 4: BUILD ERROR BLACKLIST
# =========================================================

error_files = set()
with open(error_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if parts and parts[0].strip():
            bad_file = os.path.basename(parts[0].replace("\\", "/").strip())
            error_files.add(bad_file)

print(f"Blacklisted {len(error_files)} corrupted files\n")

# =========================================================
# STEP 5: READ meta.txt -> class mapping
# =========================================================

class_files = defaultdict(list)
with open(meta_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            filename = os.path.basename(parts[0].replace("\\", "/").strip())
            label = parts[1].strip()

            if filename not in error_files:
                class_files[label].append(filename)

if not class_files:
    raise ValueError("No class/file mappings found in meta.txt")

# =========================================================
# STEP 6: PRINT CLASS DISTRIBUTION
# =========================================================

print(f"{'Class':<28} {'Total':>6}  {'Train':>6}  {'Val':>5}  {'Test':>5}")
print("-" * 58)

for label in sorted(class_files):
    n = len(class_files[label])
    tr = int(n * TRAIN_RATIO)
    va = int(n * VAL_RATIO)
    te = n - tr - va
    print(f"{label:<28} {n:>6}  {tr:>6}  {va:>5}  {te:>5}")

# =========================================================
# STEP 7: CLEAN OLD OUTPUT (optional)
# =========================================================

if os.path.exists(OUTPUT_DIR):
    print(f"\nRemoving old output folder:\n{OUTPUT_DIR}\n")
    shutil.rmtree(OUTPUT_DIR)

# =========================================================
# STEP 8: SPLIT AND COPY FILES
# =========================================================

total_counts = {"train": 0, "val": 0, "test": 0}
missing_files = []

for label, files in class_files.items():
    safe_label = label.replace("/", "_").replace(" ", "_")
    random.shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:]
    }

    for split_name, split_files in splits.items():
        out_dir = os.path.join(OUTPUT_DIR, split_name, safe_label)
        os.makedirs(out_dir, exist_ok=True)

        copied = 0
        for fname in split_files:
            src = audio_index.get(fname)
            dst = os.path.join(out_dir, fname)

            if src and os.path.exists(src):
                shutil.copy2(src, dst)
                copied += 1
            else:
                missing_files.append(fname)

        total_counts[split_name] += copied

# =========================================================
# STEP 9: FINAL SUMMARY
# =========================================================

print("\n" + "=" * 58)
print("Final file counts across all classes:")
for split_name, count in total_counts.items():
    print(f"{split_name:<8} {count} files")

expected_total = sum(len(v) for v in class_files.values())
copied_total = sum(total_counts.values())

print(f"\nExpected usable files from meta.txt: {expected_total}")
print(f"Actually copied files:              {copied_total}")
print(f"\nOrganized dataset saved to:\n{OUTPUT_DIR}")

if missing_files:
    print(f"\nWarning: {len(missing_files)} files were listed in meta.txt but not found in extracted audio.")
    print("First 10 missing files:")
    for f in missing_files[:10]:
        print("  ", f)
else:
    print("\nAll files copied successfully.")