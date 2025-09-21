import cv2
import face_recognition
import pickle
import os

ENCODINGS_PATH = os.path.join(os.path.dirname(__file__), '..', 'encodings.pkl')


def register_face(username):
    print("[DEBUG] Attempting to open camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera. Make sure it is connected and not used by another application.")
        import sys
        sys.exit(1)
    print(f"[INFO] Camera opened successfully. Please look at the camera, {username}. Press 'c' to capture.")
    encoding = None
    window_shown = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from camera.")
            continue
        if not window_shown:
            print("[DEBUG] Showing camera window now.")
            window_shown = True
        cv2.imshow("Register Face - Press 'c' to capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb_frame)
            if len(boxes) == 1:
                encoding = face_recognition.face_encodings(rgb_frame, boxes)[0]
                print("[INFO] Face captured and encoded.")
                break
            elif len(boxes) == 0:
                print("[WARN] No face detected. Try again.")
            else:
                print("[WARN] Multiple faces detected. Only one person should be in the frame.")
        elif key == ord('q'):
            print("[INFO] Registration cancelled.")
            break
    cap.release()
    cv2.destroyAllWindows()
    if encoding is not None:
        if os.path.exists(ENCODINGS_PATH):
            with open(ENCODINGS_PATH, 'rb') as f:
                data = pickle.load(f)
        else:
            data = {}
        data[username] = encoding
        with open(ENCODINGS_PATH, 'wb') as f:
            pickle.dump(data, f)
        print(f"[INFO] Encoding for {username} saved.")
    else:
        print("[ERROR] No encoding saved.")
        import sys
        sys.exit(2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python register_face_encoding.py <username>")
    else:
        register_face(sys.argv[1])
