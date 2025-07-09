# backend/detection/face_detection.py

import cv2
from detection.analyze_facial_symmetry import analyze_symmetry

def detect_facial_droop_from_frame(frame, return_detail=False):
    """
    Mendeteksi simetri wajah dan mengkategorikan: Normal / Ringan / Sedang / Berat
    Params:
        - frame: gambar dari webcam (OpenCV)
        - return_detail: jika True, kembalikan data lengkap (delta + saran)
    Returns:
        - Jika return_detail=False: "Normal" / "Sedang" / "Berat"
        - Jika return_detail=True: dict dengan semua data
    """
    result = analyze_symmetry(frame)

    if result["status"] != "ok":
        return result if return_detail else "Tidak terdeteksi"

    if return_detail:
        return result
    else:
        return result["kategori"]
