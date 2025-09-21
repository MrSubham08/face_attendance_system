# IMPORTANT: Activate venv311 before running this app
# In PowerShell, run:
#   .\venv311\Scripts\activate
# Then install dependencies:
#   pip install streamlit streamlit-option-menu opencv-python face_recognition pandas
from streamlit_option_menu import option_menu
import streamlit as st
st.set_page_config(page_title="Face Attendance System", layout="wide")
st.title("Face Attendance System")

# --- Modern Streamlit App with Navbar ---
import cv2
import numpy as np
import face_recognition
import pandas as pd
import json
import pickle
import os
from datetime import datetime

st.set_page_config(page_title="Face Attendance System", layout="wide")

# Helper functions
STUDENTS_PATH = "students.json"
LABELS_PATH = "labels.json"
ENCODINGS_PATH = "encodings.pkl"
ATTENDANCE_CSV = "attendance.csv"

def load_students():
    if os.path.exists(STUDENTS_PATH):
        with open(STUDENTS_PATH, "r") as f:
            return json.load(f)
    return {}

def load_encodings():
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_students(students):
    with open(STUDENTS_PATH, "w") as f:
        json.dump(students, f, indent=2)

def save_encodings(encodings):
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(encodings, f)

def mark_attendance(username, full_name, branch):
    columns = ["date", "name", "full_name", "branch", "time", "status"]
    today_str = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now()
    new_row = {
        "date": today_str,
        "name": username,
        "full_name": full_name,
        "branch": branch,
        "time": now.strftime("%H:%M:%S"),
        "status": "P"
    }
    if os.path.exists(ATTENDANCE_CSV):
        df = pd.read_csv(ATTENDANCE_CSV)
    else:
        df = pd.DataFrame(columns=columns)
    if not ((df["date"] == today_str) & (df["name"] == username)).any():
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df = df[columns]
        df.to_csv(ATTENDANCE_CSV, index=False)

# --- Navbar and Section Logic ---


# --- Headline ---
st.markdown("<h1 style='text-align:center; color:#1a237e; font-size:2.8rem; font-weight:800; margin-bottom:0.5rem; margin-top:0.5rem; letter-spacing:1px;'>Face Attendance System</h1>", unsafe_allow_html=True)

# --- Advanced Navbar ---
with st.container():
    selected = option_menu(
        menu_title=None,
        options=["Home", "New Registration", "Take Attendance", "View Attendance"],
        icons=["house", "person-plus", "camera", "table"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"background-color": "#1a237e", "padding": "0.5rem 0", "border-radius": "12px", "margin-bottom": "2rem"},
            "nav-link": {"font-size": "1.2rem", "font-weight": "600", "color": "#fff", "margin": "0 2rem", "border-radius": "8px"},
            "nav-link-selected": {"background-color": "#00c853", "color": "#fff"},
        }
    )
section = selected

# --- Section Content ---
if section == "Home":
    st.subheader("Welcome to the Face Attendance System")
    st.write("Use the navigation bar above to register new users, take attendance, or view attendance records.")

elif section == "New Registration":
    st.subheader("Register New User")
    st.info("You can register a new user by capturing a photo from your device camera or by uploading an image below.")
    with st.form("register_form"):
        username = st.text_input("Username (Name or Roll Number)")
        full_name = st.text_input("Full Name")
        branch = st.text_input("Branch")
        camera_photo = st.camera_input("Take a photo with your camera")
        uploaded_image = st.file_uploader("Or upload a face image", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Register")
        if submitted:
            if not username or not full_name or not branch or (not camera_photo and not uploaded_image):
                st.error("All fields and a face image (camera or upload) are required.")
            else:
                students = load_students()
                students[username] = {"full_name": full_name, "branch": branch}
                save_students(students)
                img = None
                if camera_photo is not None:
                    img_bytes = camera_photo.getvalue()
                    img_array = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                elif uploaded_image is not None:
                    img_bytes = uploaded_image.read()
                    img_array = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_img)
                    if encodings:
                        all_encodings = load_encodings()
                        all_encodings[username] = encodings[0]
                        save_encodings(all_encodings)
                        st.success(f"User {full_name} ({username}) registered!")
                    else:
                        st.error("No face detected in the provided image.")

elif section == "Take Attendance":
    st.subheader("Take Attendance (Webcam)")
    st.info("Press 'Start Attendance' to open your webcam and mark attendance for registered users.")
    if st.button("Start Attendance"):
        students = load_students()
        encodings_data = load_encodings()
        known_encodings = list(encodings_data.values())
        known_usernames = list(encodings_data.keys())
        if not known_encodings:
            st.error("No registered users found. Please register a user first.")
        else:
            cap = cv2.VideoCapture(0)
            st.info("Press 'q' in the webcam window to quit.")
            marked_today = set()
            unknown_detected = False
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam.")
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    if True in matches:
                        best_match_index = np.argmin(face_distances)
                        username = known_usernames[best_match_index]
                        if username not in marked_today:
                            student = students.get(username, {})
                            mark_attendance(username, student.get("full_name", username), student.get("branch", ""))
                            marked_today.add(username)
                        display_name = student.get("full_name", username)
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, display_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        unknown_detected = True
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown - Please register first", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Attendance - Press 'q' to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
            if marked_today:
                st.success("Attendance marked for recognized users.")
            if unknown_detected:
                st.warning("Unknown face(s) detected. Please register first if you want to mark attendance.")

elif section == "View Attendance":
    st.subheader("View Attendance Records")
    if os.path.exists(ATTENDANCE_CSV):
        df = pd.read_csv(ATTENDANCE_CSV)
        st.dataframe(df)
        st.download_button("Export as CSV", data=df.to_csv(index=False), file_name="attendance_export.csv", mime="text/csv")
    else:
        st.info("No attendance records found.")
