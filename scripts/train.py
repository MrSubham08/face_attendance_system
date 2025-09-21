# scripts/train.py
import os
import cv2
import numpy as np
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
MODEL_PATH = os.path.join(MODEL_DIR, "lbph.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(LABELS_PATH):
    print("No labels.json found. Run collect_faces.py first to register people.")
    raise SystemExit


with open(LABELS_PATH, "r", encoding="utf-8") as f:
    name_to_id = json.load(f)  # keys: names, values: ids

faces = []
labels = []

for name, person_id in name_to_id.items():
    person_dir = os.path.join(RAW_DIR, name)
    if not os.path.isdir(person_dir):
        print(f"[WARN] No directory for {name} -> skipping.")
        continue
    for fname in os.listdir(person_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(person_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (200, 200))
        faces.append(img)
        labels.append(int(person_id))

if len(faces) == 0:
    print("No training data found. Capture faces first.")
    raise SystemExit

faces_np = np.array(faces)
labels_np = np.array(labels)

# Create recognizer and train
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces_np, labels_np)
recognizer.save(MODEL_PATH)

print(f"[OK] Training complete. Model saved to: {MODEL_PATH}")
print(f"[OK] Classes: {list(name_to_id.keys())}")
