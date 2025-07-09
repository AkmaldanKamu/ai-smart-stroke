import cv2

def detect_facial_droop_from_frame(frame):
    # Untuk demo, kita deteksi simetris wajah berdasarkan senyum (placeholder)
    # Versi final bisa pakai facial landmark + rasio pipi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "Tidak terdeteksi wajah"

    # Placeholder untuk demo
    return "Miring"  # atau "Normal"
