# backend/main.py

import cv2
from detection.face_detection import detect_facial_droop_from_frame
from detection.hand_detection import detect_arm_drift_openvino
from detection.voice_detection import detect_speech_clarity
from smart_camera_selector import find_real_camera

def main():
    print("🎥 Mendeteksi kamera utama...")
    cam_index = find_real_camera()
    if cam_index is None:
        print("❌ Tidak ada kamera tersedia.")
        return

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("❌ Gagal membuka kamera.")
        return

    print("📸 Kamera aktif. Tekan 'q' untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Gagal membaca frame.")
            break

        # Tampilkan preview
        cv2.imshow("🔍 Kamera Live - AI-SMART", frame)

        # Tekan tombol:
        # - 'f' → deteksi wajah
        # - 'h' → deteksi tangan
        # - 'v' → deteksi suara
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            print("😊 Deteksi Wajah:")
            print(detect_facial_droop_from_frame(frame, return_detail=True))
        elif key == ord('h'):
            print("🖐️ Deteksi Tangan:")
            print(detect_arm_drift_openvino(frame))
        elif key == ord('v'):
            print("🎤 Deteksi Suara:")
            print(detect_speech_clarity(return_text=True))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
