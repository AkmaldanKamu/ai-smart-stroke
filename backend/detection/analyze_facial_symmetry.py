import cv2
import mediapipe as mp
from math import atan2, degrees

mp_face = mp.solutions.face_mesh

def angle_between(p1, p2):
    return degrees(atan2(p2.y - p1.y, p2.x - p1.x))

def analyze_symmetry_plus(frame):
    face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return {
            'status': 'error',
            'message': 'Wajah tidak terdeteksi'
        }

    landmarks = results.multi_face_landmarks[0].landmark

    # Ambil titik penting
    l_bibir = landmarks[61]
    r_bibir = landmarks[291]
    l_alis = landmarks[105]
    r_alis = landmarks[334]
    l_pipi = landmarks[234]
    r_pipi = landmarks[454]
    l_mata = landmarks[159]
    r_mata = landmarks[386]

    # Hitung delta Y
    delta_bibir = abs(l_bibir.y - r_bibir.y)
    delta_alis  = abs(l_alis.y - r_alis.y)
    delta_pipi  = abs(l_pipi.y - r_pipi.y)
    delta_mata  = abs(l_mata.y - r_mata.y)

    # Hitung sudut wajah kiri–kanan
    angle = angle_between(l_pipi, r_pipi)

    # Skor gabungan (rata-rata delta + bonus jika sudut tajam)
    average_delta = (delta_bibir + delta_alis + delta_pipi + delta_mata) / 4
    bonus = abs(angle) / 90  # normalnya ~0, makin miring makin besar
    final_score = average_delta + (bonus * 0.05)

    # Penilaian
    if final_score < 0.015:
        kategori = "Normal"
        saran = "Wajah simetris. Tidak ada tanda stroke."
    elif final_score < 0.035:
        kategori = "Ringan"
        saran = "Terdapat sedikit asimetri. Perlu observasi lebih lanjut."
    elif final_score < 0.06:
        kategori = "Sedang"
        saran = "Asimetri wajah terlihat. Konsultasikan ke dokter."
    else:
        kategori = "Berat"
        saran = "Tanda jelas facial droop. Segera konsultasikan medis."

    return {
        'status': 'ok',
        'kategori': kategori,
        'saran': saran,
        'score': round(final_score, 4),
        'angle': round(angle, 2),
        'delta_bibir': round(delta_bibir, 4),
        'delta_alis': round(delta_alis, 4),
        'delta_pipi': round(delta_pipi, 4),
        'delta_mata': round(delta_mata, 4)
    }
