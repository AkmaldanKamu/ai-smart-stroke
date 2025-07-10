import cv2
from detection.analyze_symmetry_pro import analyze_symmetry_pro
from smart_camera_selector import find_real_camera

def detect_facial_droop_from_frame(frame=None, return_detail=False):
    """
    Deteksi simetri wajah dari sebuah frame.
    Jika frame tidak diberikan, otomatis akan mengambil dari kamera utama.
    
    Args:
        frame (np.ndarray, optional): Gambar/frame dari kamera.
        return_detail (bool): Jika True, return dict lengkap dari analyze_symmetry_pro.
                              Jika False, hanya return kategori.

    Returns:
        Union[str, dict]: Hasil kategori deteksi atau dict lengkap.
    """

    # Ambil frame dari kamera jika tidak ada input frame
    if frame is None:
        camera_index = find_real_camera()
        cap = cv2.VideoCapture(camera_index)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return {"status": "error", "message": "Gagal mengambil gambar dari kamera"}

    result = analyze_symmetry_pro(frame)

    # Jika gagal deteksi landmark/simetris
    if result.get("status") != "ok":
        return result if return_detail else "Tidak terdeteksi"

    return result if return_detail else result.get("kategori", "Tidak diketahui")
