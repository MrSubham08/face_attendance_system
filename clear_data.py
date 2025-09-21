import os
import json
import pickle
import pandas as pd

# Remove all registered faces
ENCODINGS_PATH = "encodings.pkl"
if os.path.exists(ENCODINGS_PATH):
    os.remove(ENCODINGS_PATH)
    print("Removed encodings.pkl (registered faces)")
else:
    print("No encodings.pkl found.")

# Clear attendance records
ATTENDANCE_CSV = "attendance.csv"
if os.path.exists(ATTENDANCE_CSV):
    df = pd.DataFrame(columns=["date", "name", "full_name", "branch", "time", "status"])
    df.to_csv(ATTENDANCE_CSV, index=False)
    print("Cleared attendance.csv (attendance records)")
else:
    print("No attendance.csv found.")
