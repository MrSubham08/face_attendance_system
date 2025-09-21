# scripts/recognize_and_log.py
import cv2
import os
import json
import pandas as pd
from datetime import datetime, date

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
MODEL_PATH = os.path.join(MODEL_DIR, "lbph.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

STUDENTS_PATH = os.path.join(BASE_DIR, "students.json")
ATTENDANCE_CSV = os.path.join(BASE_DIR, "attendance.csv")

if not os.path.exists(LABELS_PATH) or not os.path.exists(MODEL_PATH):
    print("labels.json or lbph.yml missing. Run collect_faces.py and train.py first.")
    raise SystemExit

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    name_to_id = json.load(f)  # keys: names, values: ids

# Build id_to_name mapping
id_to_name = {int(v): k for k, v in name_to_id.items()}

# Load student details
if os.path.exists(STUDENTS_PATH):
    with open(STUDENTS_PATH, "r", encoding="utf-8") as f:
        students = json.load(f)
else:
    students = {}

# Load attendance CSV or create new


# Always use the correct columns
columns = ["date", "name", "full_name", "branch", "time", "status"]
if os.path.exists(ATTENDANCE_CSV):
    df = pd.read_csv(ATTENDANCE_CSV)
    # Add missing columns if any
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    df = df[columns]
else:
    df = pd.DataFrame(columns=columns)

today_str = date.today().isoformat()



# Track who is already marked today (by name and date)
marked_today = set(df[(df["date"] == today_str)]["name"].tolist())

# Setup recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Could not open webcam.")
    raise SystemExit

THRESHOLD = 70.0  # Lower = stricter match. Tune between 50-90



print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        face = cv2.resize(roi_gray, (200, 200))
        label_id, confidence = recognizer.predict(face)  # label_id is int

        # Get the username (as string) from id_to_name
        username = id_to_name.get(label_id, None)
        if username is not None:
            student = students.get(username, {})
            full_name = student.get("full_name", username)
            branch = student.get("branch", "")
            if confidence <= THRESHOLD:
                display = f"{full_name} | {branch} ({confidence:.1f})"
            else:
                display = f"Unknown ({confidence:.1f})"
        else:
            display = f"Unknown ({confidence:.1f})"

        if username is not None and confidence <= THRESHOLD:
            now = datetime.now()
            # Only mark if not already present for today
            if username not in marked_today:
                full_name = students.get(username, {}).get("full_name", username)
                branch = students.get(username, {}).get("branch", "")
                new_row = {
                    "date": today_str,
                    "name": username,
                    "full_name": full_name,
                    "branch": branch,
                    "time": now.strftime("%H:%M:%S"),
                    "status": "P"
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df = df[columns]  # Ensure column order
                df.to_csv(ATTENDANCE_CSV, index=False)
                marked_today.add(username)
                print(f"[MARKED] {full_name} ({username}) at {new_row['time']}")

        # Draw on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.putText(frame, display, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Attendance - q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Session ended. Attendance saved to {ATTENDANCE_CSV}")
