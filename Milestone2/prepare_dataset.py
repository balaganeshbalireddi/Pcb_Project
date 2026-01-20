import os
import shutil
import random

RAW = "data/raw_rois"
TRAIN = "data/train"
TEST = "data/test"

classes = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spurious_copper",
    "spur"
]

for c in classes:
    os.makedirs(os.path.join(TRAIN, c), exist_ok=True)
    os.makedirs(os.path.join(TEST, c), exist_ok=True)

# Group files by class
class_files = {c: [] for c in classes}

for f in os.listdir(RAW):
    name = f.lower()
    for c in classes:
        if c in name:
            class_files[c].append(f)
            break

# Split safely
for c, files in class_files.items():
    random.shuffle(files)

    split_idx = int(len(files) * 0.8)

    train_files = files[:split_idx]
    test_files = files[split_idx:]

    # Ensure at least 1 in test
    if len(test_files) == 0 and len(train_files) > 1:
        test_files.append(train_files.pop())

    for f in train_files:
        shutil.copy(os.path.join(RAW, f), os.path.join(TRAIN, c, f))

    for f in test_files:
        shutil.copy(os.path.join(RAW, f), os.path.join(TEST, c, f))

print("🎯 Safe dataset split completed")
