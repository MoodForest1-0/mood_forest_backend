import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace

DATASET_DIR = "dataset"

EMOTIONS = ['angry', 'fear', 'happy', 'neutral', 'sad']

embeddings = []
labels = []

print("🚀 Starting embedding extraction...")

for emotion in EMOTIONS:
    folder_path = os.path.join(DATASET_DIR, emotion)

    for img_name in tqdm(os.listdir(folder_path), desc=emotion):
        img_path = os.path.join(folder_path, img_name)

        # ✅ SKIP HEIC FILES HERE (correct place)
        if img_path.lower().endswith(".heic"):
            print("Skipping HEIC:", img_path)
            continue

        try:
            result = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet512",
                enforce_detection=False
            )

            embedding = result[0]["embedding"]

            embeddings.append(embedding)
            labels.append(emotion)

        except Exception as e:
            print("Error with", img_path, ":", e)

# Convert to numpy arrays
X = np.array(embeddings)
y = np.array(labels)

print("\n✅ Extraction complete")
print("Total samples:", len(X))
print("Feature size:", X.shape)

# Save to disk
np.save("embeddings/X.npy", X)
np.save("embeddings/y.npy", y)

print("💾 Saved to embeddings/ folder")