# backend/detection/face_detection.py

import cv2
from detection.analyze_facial_symmetry import analyze_symmetry_plus

def detect_facial_droop_from_frame(frame, return_detail=False):
    result = analyze_symmetry_plus(frame)
    if result["status"] != "ok":
        return result if return_detail else "Tidak terdeteksi"
    return result if return_detail else result["kategori"]
