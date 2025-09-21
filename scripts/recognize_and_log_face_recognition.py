import cv2
import face_recognition
import pickle
import os
import pandas as pd
from datetime import datetime

ENCODINGS_PATH = os.path.join(os.path.dirname(__file__), '..', 'encodings.pkl')
STUDENTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'students.json')
ATTENDANCE_CSV = os.path.join(os.path.dirname(__file__), '..', 'attendance.csv')

# Load encodings and students
with open(ENCODINGS_PATH, 'rb') as f:
    encodings_data = pickle.load(f)
import json
with open(STUDENTS_PATH, 'r') as f:
    students = json.load(f)

known_encodings = list(encodings_data.values())
known_usernames = list(encodings_data.keys())

# Prepare attendance DataFrame
columns = ["date", "name", "full_name", "branch", "time", "status"]
if os.path.exists(ATTENDANCE_CSV):
    df = pd.read_csv(ATTENDANCE_CSV)
else:
    df = pd.DataFrame(columns=columns)

today_str = datetime.now().strftime("%Y-%m-%d")
marked_today = set(df[df['date'] == today_str]['name']) if not df.empty else set()

cap = cv2.VideoCapture(0)
print("[INFO] Press 'q' to quit.")
window_name = "Attendance - q to quit"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
fail_count = 0
max_failures = 30  # About 1 second if running at ~30fps
while True:
    ret, frame = cap.read()
    if not ret:
        fail_count += 1
        print(f"[WARN] Can't grab frame from camera. Attempt {fail_count}/{max_failures}")
        if fail_count >= max_failures:
            print("[ERROR] Camera not available. Exiting attendance.")
            break
        continue
    fail_count = 0
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        name = "Unknown"
        display = "Unknown"
        if True in matches:
            best_match_index = face_distances.argmin()
            username = known_usernames[best_match_index]
            student = students.get(username, {})
            full_name = student.get("full_name", username)
            branch = student.get("branch", "")
            name = username
            display = f"{full_name} | {branch}"
            # Only mark attendance for known faces (not 'Unknown')
            if name != "Unknown" and name not in marked_today:
                now = datetime.now()
                new_row = {
                    "date": today_str,
                    "name": name,
                    "full_name": full_name,
                    "branch": branch,
                    "time": now.strftime("%H:%M:%S"),
                    "status": "P"
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df = df[columns]
                df.to_csv(ATTENDANCE_CSV, index=False)
                marked_today.add(name)
                print(f"[MARKED] {full_name} ({name}) at {new_row['time']}")
        else:
            # Draw a filled rectangle at the bottom of the frame
            msg = "Face not recognized. Please register first."
            h, w = frame.shape[:2]
            rect_height = 40
            cv2.rectangle(frame, (0, h - rect_height), (w, h), (255, 255, 255), -1)
            # Calculate text size and position for centering
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_size, _ = cv2.getTextSize(msg, font, font_scale, thickness)
            text_x = (w - text_size[0]) // 2
            text_y = h - (rect_height // 2) + (text_size[1] // 2)
            cv2.putText(frame, msg, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, display, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
