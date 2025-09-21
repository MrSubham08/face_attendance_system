import cv2
import os
import json


# === Configuration ===
DATA_DIR = os.path.join("data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)
LABELS_PATH = "labels.json"




# Ask for student's details
while True:
    student_name = input("Enter student username (number): ").strip()
    if student_name.isdigit():
        break
    print("Username must be a number. Please try again.")
full_name = input("Enter full name: ").strip()
branch = input("Enter branch: ").strip()
student_folder = os.path.join(DATA_DIR, student_name)
os.makedirs(student_folder, exist_ok=True)

# Save student details in students.json
STUDENTS_PATH = "students.json"
if os.path.exists(STUDENTS_PATH):
    with open(STUDENTS_PATH, "r") as f:
        students = json.load(f)
else:
    students = {}
students[student_name] = {"full_name": full_name, "branch": branch}
with open(STUDENTS_PATH, "w") as f:
    json.dump(students, f, indent=2)

# Load or create labels.json
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
else:
    labels = {}

# Assign a new label if student not present
if student_name not in labels:
    if labels:
        next_label = max(labels.values()) + 1
    else:
        next_label = 0
    labels[student_name] = next_label
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
print("Press 'q' to quit collecting faces.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        file_path = os.path.join(student_folder, f"{count}.jpg")
        cv2.imwrite(file_path, face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Collecting Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= 50:  # Stop after 50 samples
        print("50 images collected. Stopping.")
        break

cap.release()
cv2.destroyAllWindows()
print(f"Images saved in: {student_folder}")
print(f"labels.json updated: {labels}")
