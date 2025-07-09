# backend/detection/nihss_scoring.py

def score_nihss(face_result, hand_result, voice_result):
    score = 0
    keterangan = []

    # Facial Droop (1A)
    if face_result.lower() == "miring":
        score += 1
        keterangan.append("Wajah miring (1 poin)")

    # Arm Drift (1B)
    if "kedua tangan turun" in hand_result.lower():
        score += 2
        keterangan.append("Kedua tangan turun (2 poin)")
    elif "tangan" in hand_result.lower() and "turun" in hand_result.lower():
        score += 1
        keterangan.append("Satu tangan turun (1 poin)")

    # Speech (1C)
    if voice_result.lower() == "tidak ada suara":
        score += 2
        keterangan.append("Tidak ada suara (2 poin)")
    elif voice_result.lower() == "suara tidak jelas":
        score += 1
        keterangan.append("Suara tidak jelas (1 poin)")

    # Interpretasi
    if score == 0:
        kategori = "Normal"
        saran = "Pasien tidak menunjukkan tanda stroke."
    elif score <= 2:
        kategori = "Ringan"
        saran = "Amati, bisa konsultasi dokter."
    elif score <= 4:
        kategori = "Sedang"
        saran = "Segera ke rumah sakit."
    else:
        kategori = "Berat"
        saran = "Panggil ambulans secepatnya!"

    return {
        "score": score,
        "kategori": kategori,
        "rincian": keterangan,
        "saran": saran
    }
    
def generate_diagnosis_summary(face_result, hand_result, voice_result, scoring):
    summary = []

    # Interpretasi Wajah
    if face_result.lower() == "miring":
        summary.append("Terjadi asimetri wajah saat diminta tersenyum, menunjukkan kemungkinan facial palsy.")
    else:
        summary.append("Tidak ditemukan kelainan pada otot wajah.")

    # Interpretasi Tangan
    if "kedua tangan turun" in hand_result.lower():
        summary.append("Kedua tangan tidak dapat diangkat sejajar, mengindikasikan kelemahan otot bilateral.")
    elif "turun" in hand_result.lower():
        summary.append("Salah satu tangan tidak bisa diangkat sejajar, menunjukkan kelemahan otot satu sisi tubuh.")
    else:
        summary.append("Kedua tangan dapat diangkat sejajar.")

    # Interpretasi Suara
    if voice_result.lower() == "tidak ada suara":
        summary.append("Pasien tidak dapat berbicara saat diminta, gejala afasia berat.")
    elif voice_result.lower() == "suara tidak jelas":
        summary.append("Ucapan terdengar tidak jelas, indikasi awal gangguan bicara (dysarthria).")
    else:
        summary.append("Ucapan terdengar normal.")

    # Tambahkan hasil akhir dari scoring
    summary.append(f"Kategori stroke: {scoring['kategori']} ({scoring['score']} poin).")
    summary.append(f"Saran tindakan: {scoring['saran']}")

    return summary
