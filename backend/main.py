import cv2
from detection.face_detection import detect_facial_droop_from_frame
from detection.hand_detection import detect_arm_drift_openvino
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

    # Deteksi wajah (simetri wajah)
    face_result = detect_facial_droop_from_frame(frame)

    # Deteksi tangan menggunakan model OpenVINO
    hand_result_obj = detect_arm_drift_openvino()
    hand_result = hand_result_obj.get('kategori', 'Tidak diketahui')

    # Tampilkan hasil deteksi di layar
    cv2.putText(frame, f"Wajah: {face_result}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Tangan: {hand_result}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("AI-SMART - Deteksi Stroke", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
