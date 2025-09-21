
import cv2
import os
import json
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox
import face_recognition


# Basic Tkinter window for Face Attendance System

def show_registration_form():
    reg_win = tk.Toplevel()
    reg_win.title("Register New User")
    reg_win.geometry("350x250")

    tk.Label(reg_win, text="Username (Name or Roll Number):").pack(pady=5)
    username_entry = tk.Entry(reg_win)
    username_entry.pack(pady=5)

    tk.Label(reg_win, text="Full Name:").pack(pady=5)
    fullname_entry = tk.Entry(reg_win)
    fullname_entry.pack(pady=5)

    tk.Label(reg_win, text="Branch:").pack(pady=5)
    branch_entry = tk.Entry(reg_win)
    branch_entry.pack(pady=5)

    from tkinter import ttk
    def submit():
        username = username_entry.get().strip()
        fullname = fullname_entry.get().strip()
        branch = branch_entry.get().strip()
        if not username or not username.replace(" ","").isalnum():
            messagebox.showerror("Error", "Username must be a non-empty name or roll number (letters/numbers only).")
            return
        if not fullname or not branch:
            messagebox.showerror("Error", "Full name and branch are required.")
            return

        # Save to students.json
        students_path = "students.json"
        if os.path.exists(students_path):
            with open(students_path, "r") as f:
                students = json.load(f)
        else:
            students = {}
        students[username] = {"full_name": fullname, "branch": branch}
        with open(students_path, "w") as f:
            json.dump(students, f, indent=2)

        # Save to labels.json
        labels_path = "labels.json"
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                labels = json.load(f)
        else:
            labels = {}
        if username not in labels:
            next_label = max(labels.values(), default=-1) + 1
            labels[username] = next_label
            with open(labels_path, "w") as f:
                json.dump(labels, f, indent=2)

        # Show progress/loading indicator
        progress_win = tk.Toplevel(reg_win)
        progress_win.title("Registering Face")
        progress_win.geometry("300x100")
        progress_win.transient(reg_win)
        progress_win.grab_set()
        tk.Label(progress_win, text="Capturing and encoding face...", font=("Segoe UI", 12)).pack(pady=15)
        pb = ttk.Progressbar(progress_win, mode="indeterminate")
        pb.pack(pady=10, fill=tk.X, padx=30)
        pb.start(10)

        reg_win.update_idletasks()

        # Register face encoding using the new script
        messagebox.showinfo("Face Registration", f"Now the system will capture your face. Please look at the camera and press 'c' to capture.")
        result = subprocess.run([sys.executable, "scripts/register_face_encoding.py", username], capture_output=True, text=True)

        pb.stop()
        progress_win.destroy()

        if result.returncode == 0:
            messagebox.showinfo("Registered", f"User {fullname} ({username}) registered and face encoding saved!")
            reg_win.destroy()
        else:
            messagebox.showerror("Error", f"Face encoding failed or was cancelled.\n\nError details:\n{result.stderr}")

    tk.Button(reg_win, text="Register", command=submit).pack(pady=15)


def train_model():
    try:
        result = subprocess.run([sys.executable, "scripts/train.py"], capture_output=True, text=True)
        if result.returncode == 0:
            messagebox.showinfo("Success", "Model trained successfully!\n" + result.stdout)
        else:
            messagebox.showerror("Error", "Training failed:\n" + result.stderr)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def run_attendance():
    import pickle
    import pandas as pd
    from datetime import datetime
    try:
        # Show progress/loading indicator
        progress_win = tk.Toplevel()
        progress_win.title("Taking Attendance")
        progress_win.geometry("300x100")
        progress_win.transient()
        progress_win.grab_set()
        tk.Label(progress_win, text="Matching faces and marking attendance...", font=("Segoe UI", 12)).pack(pady=15)
        from tkinter import ttk
        pb = ttk.Progressbar(progress_win, mode="indeterminate")
        pb.pack(pady=10, fill=tk.X, padx=30)
        pb.start(10)

        # Attendance logic (integrated)
        ENCODINGS_PATH = "encodings.pkl"
        STUDENTS_PATH = "students.json"
        ATTENDANCE_CSV = "attendance.csv"
        with open(ENCODINGS_PATH, 'rb') as f:
            encodings_data = pickle.load(f)
        with open(STUDENTS_PATH, 'r') as f:
            students = json.load(f)
        known_encodings = list(encodings_data.values())
        known_usernames = list(encodings_data.keys())
        columns = ["date", "name", "full_name", "branch", "time", "status"]
        if os.path.exists(ATTENDANCE_CSV):
            df = pd.read_csv(ATTENDANCE_CSV)
        else:
            df = pd.DataFrame(columns=columns)
        today_str = datetime.now().strftime("%Y-%m-%d")
        marked_today = set(df[df['date'] == today_str]['name']) if not df.empty else set()

        pb.stop()
        progress_win.destroy()

        # Open camera window
        cap = cv2.VideoCapture(0)
        window_name = "Attendance - q to quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        fail_count = 0
        max_failures = 30
        while True:
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                if fail_count >= max_failures:
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
                    if name != "Unknown" and name not in marked_today:
                        # Exclude test/dummy users from attendance
                        if name.lower().startswith("test") or name.lower().startswith("dummy"):
                            continue
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
                else:
                    msg = "Face not recognized. Please register first."
                    h, w = frame.shape[:2]
                    rect_height = 40
                    cv2.rectangle(frame, (0, h - rect_height), (w, h), (255, 255, 255), -1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    text_size, _ = cv2.getTextSize(msg, font, font_scale, thickness)
                    text_x = (w - text_size[0]) // 2
                    text_y = h - (rect_height // 2) + (text_size[1] // 2)
                    cv2.putText(frame, msg, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, display, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        messagebox.showerror("Error", str(e))


import csv
import json


def view_attendance():
    viewer = tk.Toplevel()
    viewer.title("Attendance Records")
    viewer.geometry("700x400")

    frame = tk.Frame(viewer)
    frame.pack(fill=tk.BOTH, expand=True)

    # Load students.json for full name and branch lookup
    students = {}
    try:
        with open("students.json", "r", encoding="utf-8") as f:
            students = json.load(f)
    except Exception:
        pass

    # Read attendance.csv and build new records with full_name and branch
    records = []
    try:
        with open("attendance.csv", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                username = row.get("name", "")
                # Prefer full_name/branch from CSV, else fetch from students.json
                full_name = row.get("full_name") or students.get(str(username), {}).get("full_name", "")
                branch = row.get("branch") or students.get(str(username), {}).get("branch", "")
                records.append([
                    row.get("date", ""),
                    full_name,
                    branch,
                    row.get("time", ""),
                    row.get("status", "")
                ])
    except Exception as e:
        tk.Label(frame, text=f"Error reading attendance.csv: {e}").pack()
        return

    if not records:
        tk.Label(frame, text="No attendance records found.").pack()
        return

    # Export buttons
    def export_csv():
        import shutil
        try:
            shutil.copyfile("attendance.csv", "attendance_export.csv")
            messagebox.showinfo("Export", "Attendance exported as attendance_export.csv")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def export_pdf():
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            pdf_file = "attendance_export.pdf"
            c = canvas.Canvas(pdf_file, pagesize=letter)
            width, height = letter
            c.setFont("Helvetica-Bold", 14)
            c.drawString(30, height - 40, "Attendance Records")
            c.setFont("Helvetica", 10)
            y = height - 70
            headers = ["date", "full_name", "branch", "time", "status"]
            c.drawString(30, y, " | ".join(headers))
            y -= 20
            for row in records:
                c.drawString(30, y, " | ".join(str(val) for val in row))
                y -= 15
                if y < 40:
                    c.showPage()
                    y = height - 40
            c.save()
            messagebox.showinfo("Export", f"Attendance exported as {pdf_file}")
        except ImportError:
            messagebox.showerror("Export Error", "reportlab not installed. Run 'pip install reportlab' in your venv.")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    btn_frame = tk.Frame(viewer)
    btn_frame.pack(pady=8)
    tk.Button(btn_frame, text="Export as CSV", command=export_csv, bg="#00c853", fg="#fff", font=("Segoe UI", 10, "bold"), relief="flat").pack(side=tk.LEFT, padx=8)
    tk.Button(btn_frame, text="Export as PDF", command=export_pdf, bg="#1a237e", fg="#fff", font=("Segoe UI", 10, "bold"), relief="flat").pack(side=tk.LEFT, padx=8)

    # Table headers
    headers = ["date", "full_name", "branch", "time", "status"]
    for j, val in enumerate(headers):
        e = tk.Entry(frame, width=18, fg='black', font=('Arial', 10, 'bold'))
        e.grid(row=0, column=j)
        e.insert(tk.END, val)
        e.config(state='readonly')

    # Display as table
    for i, row in enumerate(records):
        for j, val in enumerate(row):
            e = tk.Entry(frame, width=18, fg='black')
            e.grid(row=i+1, column=j)
            e.insert(tk.END, val)
            e.config(state='readonly')

def main():
    root = tk.Tk()
    root.title("Face Attendance System")
    root.state('zoomed')  # Start maximized
    root.configure(bg="#f9f9f9")

    # Header with a modern logo
    header = tk.Frame(root, bg="#1a237e", height=120)
    header.pack(fill=tk.X)
    logo = tk.Canvas(header, width=80, height=80, bg="#1a237e", highlightthickness=0)
    # Green accent circle
    logo.create_oval(10, 10, 70, 70, fill="#00c853", outline="")
    # White face icon (simple vector)
    logo.create_oval(25, 30, 55, 60, fill="#fff", outline="#fff")  # face
    logo.create_oval(33, 42, 37, 46, fill="#1a237e", outline="")  # left eye
    logo.create_oval(43, 42, 47, 46, fill="#1a237e", outline="")  # right eye
    logo.create_arc(35, 50, 45, 58, start=200, extent=140, style='arc', outline="#1a237e", width=2)
    logo.pack(side=tk.LEFT, padx=30, pady=20)
    tk.Label(header, text="Face Attendance System", font=("Segoe UI", 32, "bold"), fg="#fff", bg="#1a237e").pack(side=tk.LEFT, padx=10, pady=20)

    # Main frame with section title, centered
    main_frame = tk.Frame(root, bg="#f9f9f9", highlightthickness=0)
    main_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    tk.Label(main_frame, text="Welcome! Please select an option:", font=("Segoe UI", 18, "bold"), bg="#f9f9f9", fg="#1a237e").pack(pady=(10, 30))

    btn_style = {"font": ("Segoe UI", 15, "bold"), "bg": "#00c853", "fg": "#fff", "activebackground": "#388e3c", "activeforeground": "#fff", "bd": 0, "relief": "flat", "width": 30, "height": 2, "cursor": "hand2"}

    tk.Button(main_frame, text="Register New User", command=show_registration_form, **btn_style).pack(pady=12)
    tk.Button(main_frame, text="Train Model", command=train_model, **btn_style).pack(pady=12)
    tk.Button(main_frame, text="Take Attendance", command=run_attendance, **btn_style).pack(pady=12)
    tk.Button(main_frame, text="View Attendance Records", command=view_attendance, **btn_style).pack(pady=12)

    # Footer
    footer = tk.Frame(root, bg="#1a237e", height=40)
    footer.pack(fill=tk.X, side=tk.BOTTOM)
    tk.Label(footer, text="© 2025 Face Attendance | Developed by Subham Kumar", font=("Segoe UI", 11), fg="#00c853", bg="#1a237e").pack(pady=8)

    root.mainloop()

if __name__ == "__main__":
    main()
