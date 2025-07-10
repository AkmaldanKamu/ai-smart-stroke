import cv2
from detection.face_detection import detect_facial_droop_from_frame
from detection.hand_detection import detect_arm_drift_from_frame

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera tidak tersedia")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_result = detect_facial_droop_from_frame(frame)
    hand_result = detect_arm_drift_from_frame(frame)

    cv2.putText(frame, f"Wajah: {face_result}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Tangan: {hand_result}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("AI-SMART - Deteksi Stroke", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
