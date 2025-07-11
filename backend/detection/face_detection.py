from detection.analyze_symmetry_pro import analyze_symmetry_pro

def detect_facial_droop_from_frame(frame, return_detail=False):
    """
    Deteksi facial droop menggunakan frame gambar (OpenCV).
    """
    
    result = analyze_symmetry_pro(frame)
    if result["status"] != "ok":
        return result if return_detail else "Tidak terdeteksi"
    return result if return_detail else result["kategori"]
