from smart_camera_selector import find_real_camera
import cv2

def detect_facial_droop_from_frame(frame=None, return_detail=False):
    if frame is None:
        cap = cv2.VideoCapture(find_real_camera())
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return {"status": "error", "message": "Gagal akses kamera"}
    
    # ... (deteksi landmark & analisis simetri)
