from PIL import Image
import numpy as np
import os

DATASET_DIR = "../DOP_data"
BLACK_THRESHOLD = 5
MAX_BLACK_RATIO = 0.95

removed = 0

for root, _, files in os.walk(DATASET_DIR):
    for fname in files:
        if not fname.lower().endswith(".png"):
            continue

        path = os.path.join(root, fname)
        img = np.array(Image.open(path))

        black_ratio = np.mean(img < BLACK_THRESHOLD)

        if black_ratio > MAX_BLACK_RATIO:
            os.remove(path)
            removed += 1

print(f"Removed {removed} black / no-data images.")
