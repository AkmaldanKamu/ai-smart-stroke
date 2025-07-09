import cv2
import mediapipe as mp

mp_face = mp.solutions.face_mesh

def analyze_symmetry(frame):
    face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return {
            'status': 'error',
            'message': 'Wajah tidak terdeteksi'
        }

    landmarks = results.multi_face_landmarks[0].landmark

    # Landmark bibir kiri & kanan (MediaPipe index)
    left_mouth = landmarks[61]   # bibir kiri
    right_mouth = landmarks[291] # bibir kanan

    # Landmark alis kiri & kanan
    left_brow = landmarks[105]
    right_brow = landmarks[334]

    delta_mouth = abs(left_mouth.y - right_mouth.y)
    delta_brow = abs(left_brow.y - right_brow.y)

    avg_delta = (delta_mouth + delta_brow) / 2

    # Kategori
    if avg_delta < 0.015:
        kategori = "Normal"
        saran = "Wajah simetris. Tidak ada tanda stroke."
    elif avg_delta < 0.03:
        kategori = "Ringan"
        saran = "Terdapat sedikit asimetri. Perlu observasi lebih lanjut."
    elif avg_delta < 0.06:
        kategori = "Sedang"
        saran = "Asimetri wajah terlihat. Konsultasikan ke dokter."
    else:
        kategori = "Berat"
        saran = "Tanda jelas facial droop. Segera konsultasikan medis."

    return {
        'status': 'ok',
        'delta_mouth': round(delta_mouth, 4),
        'delta_brow': round(delta_brow, 4),
        'score': round(avg_delta, 4),
        'kategori': kategori,
        'saran': saran
    }
