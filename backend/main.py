import cv2
from detection.face_detection import detect_facial_droop_from_frame
from smart_camera_selector import find_real_camera

# Gunakan kamera utama (hindari Iriun/virtual)
camera_index = find_real_camera()
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("❌ Kamera tidak tersedia")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi wajah
    face_result = detect_facial_droop_from_frame(frame)

    # Tampilkan informasi awal
    cv2.putText(frame, f"Wajah: {face_result}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("AI-SMART - Deteksi Stroke", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
