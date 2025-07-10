import cv2

def find_available_cameras(max_index=5):
    print("🔍 Mendeteksi kamera yang tersedia...")
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Kamera ditemukan di index {i}")
            cap.release()
        else:
            print(f"❌ Tidak ada kamera di index {i}")

find_available_cameras()