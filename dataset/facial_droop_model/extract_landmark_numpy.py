import deeplake
import numpy as np
import os
import cv2

# Load dataset
ds = deeplake.load("hub://activeloop/300w")

X = []
y = []

SAVE_PATH = "facial_droop_model/landmarks"
os.makedirs(SAVE_PATH, exist_ok=True)

print("📦 Mengekstrak landmark dari dataset...")

for i in range(len(ds)):
    try:
        img = ds[i]['images'].numpy()
        landmarks = ds[i]['landmarks'].numpy()  # shape (68, 2)

        # Landmark asli (normal)
        flat = landmarks.flatten()
        X.append(flat)
        y.append(0)  # Label: normal

        # Landmark hasil transformasi (simulasi stroke)
        skewed = landmarks.copy()
        skewed[:, 0] += 5  # Miringkan ke kanan (simulasi stroke)
        X.append(skewed.flatten())
        y.append(1)  # Label: miring

    except Exception as e:
        print(f"⚠️ Error di index {i}: {e}")
        continue

X = np.array(X)
y = np.array(y)

np.save(os.path.join(SAVE_PATH, "features.npy"), X)
np.save(os.path.join(SAVE_PATH, "labels.npy"), y)

print(f"✅ Selesai! Total sample: {len(X)}")
