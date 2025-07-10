# backend/smart_camera_selector.py

import cv2

def find_real_camera(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None and frame.shape[0] > 100:
                print(f"✅ Kamera utama ditemukan di index {i}")
                return i
    print("❌ Tidak ada kamera valid ditemukan.")
    return 0  # fallback default
