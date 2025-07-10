import cv2
import mediapipe as mp
from math import atan2, degrees

mp_face = mp.solutions.face_mesh

def angle_between(p1, p2):
    return degrees(atan2(p2.y - p1.y, p2.x - p1.x))

def analyze_symmetry_pro(frame):
    face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return {
            'status': 'error',
            'message': 'Wajah tidak terdeteksi'
        }

    landmarks = results.multi_face_landmarks[0].landmark

    # --- BIBIR ---
    l_bibir = landmarks[61]
    r_bibir = landmarks[291]
    mid_bibir = landmarks[13]

    delta_bibir_y = abs(l_bibir.y - r_bibir.y)
    ratio_bibir_x = abs(l_bibir.x - r_bibir.x)

    # --- MATA ---
    top_l_eye = landmarks[159]
    bot_l_eye = landmarks[145]
    top_r_eye = landmarks[386]
    bot_r_eye = landmarks[374]

    l_eye_openness = abs(top_l_eye.y - bot_l_eye.y)
    r_eye_openness = abs(top_r_eye.y - bot_r_eye.y)
    delta_mata = abs(l_eye_openness - r_eye_openness)
    ptosis_suspect = delta_mata > 0.015

    # --- PIPI / HEAD TILT ---
    l_pipi = landmarks[234]
    r_pipi = landmarks[454]
    angle_face = angle_between(l_pipi, r_pipi)

    # --- Penilaian per komponen ---
    nilai_bibir = 0 if delta_bibir_y < 0.01 else 1 if delta_bibir_y < 0.03 else 2
    nilai_mata = 0 if delta_mata < 0.01 else 1 if delta_mata < 0.02 else 2
    nilai_pipi = 0 if abs(angle_face) < 3 else 1 if abs(angle_face) < 6 else 2

    nilai_total = nilai_bibir + nilai_mata + nilai_pipi

    # --- Kategori Final ---
    if nilai_total <= 1:
        kategori = "Normal"
        saran = "Wajah simetris. Tidak ada tanda stroke."
    elif nilai_total <= 3:
        kategori = "Ringan"
        saran = "Terdapat sedikit asimetri. Perlu observasi."
    elif nilai_total <= 5:
        kategori = "Sedang"
        saran = "Asimetri wajah terlihat. Konsultasikan ke dokter."
    else:
        kategori = "Berat"
        saran = "Tanda jelas facial droop. Segera periksa medis."

    return {
        'status': 'ok',
        'kategori': kategori,
        'saran': saran,
        'score_total': nilai_total,
        'penilaian': {
            'bibir': {
                'delta_y': round(delta_bibir_y, 4),
                'rasio_x': round(ratio_bibir_x, 4),
                'nilai': nilai_bibir
            },
            'mata': {
                'delta_openness': round(delta_mata, 4),
                'ptosis_suspect': ptosis_suspect,
                'nilai': nilai_mata
            },
            'pipi': {
                'kemiringan_derajat': round(angle_face, 2),
                'nilai': nilai_pipi
            }
        }
    }
