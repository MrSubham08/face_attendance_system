# main.py
import subprocess
import sys

def run_script(path):
    subprocess.run([sys.executable, path])

def menu():
    while True:
        print("\nFace Attendance - Menu")
        print("1) Collect faces (open camera)")
        print("2) Train model")
        print("3) Run attendance (recognize)")
        print("4) Exit")
        choice = input("Choose: ").strip()
        if choice == "1":
            run_script("scripts/collect_faces.py")
        elif choice == "2":
            run_script("scripts/train.py")
        elif choice == "3":
            run_script("scripts/recognize_and_log.py")
        elif choice == "4":
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    menu()
