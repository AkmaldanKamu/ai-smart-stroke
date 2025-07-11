# backend/detection/nihss_scoring.py

def score_nihss(face_score, voice_result):
    """
    Menghitung skor NIHSS berdasarkan skor wajah dan hasil suara.
    face_score: integer dari 0–6
    voice_result: string ('Suara jelas', 'Suara tidak jelas', 'Tidak ada suara')
    """
    score = 0
    rincian = []

    # ✅ Skor wajah langsung ditambahkan
    score += face_score
    rincian.append(f"Ekspresi wajah (skor: {face_score}/6)")

    # ✅ Skor suara
    voice_result = voice_result.lower().strip()
    if "tidak ada suara" in voice_result:
        score += 2
        rincian.append("Tidak ada suara (2 poin)")
    elif "tidak jelas" in voice_result:
        score += 1
        rincian.append("Suara tidak jelas (1 poin)")
    else:
        rincian.append("Suara normal (0 poin)")

    # ✅ Interpretasi total
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
        'score': score,
        'kategori': kategori,
        'rincian': rincian,
        'saran': saran
    }

def generate_diagnosis_summary(face_kategori, voice_result, scoring):
    """
    Ringkasan interpretasi untuk user berdasarkan input dan hasil scoring.
    """
    summary = []

    # Interpretasi wajah
    if face_kategori.lower() == "normal":
        summary.append("Tidak ditemukan kelainan pada otot wajah.")
    else:
        summary.append(f"Wajah menunjukkan gejala: {face_kategori}.")

    # Interpretasi suara
    voice = voice_result.lower()
    if "tidak ada suara" in voice:
        summary.append("Pasien tidak dapat berbicara saat diminta, gejala afasia berat.")
    elif "tidak jelas" in voice:
        summary.append("Ucapan terdengar tidak jelas, indikasi gangguan bicara (dysarthria).")
    else:
        summary.append("Ucapan terdengar normal.")

    # Kesimpulan dan saran
    summary.append(f"Kategori stroke: {scoring['kategori']} ({scoring['score']} poin).")
    summary.append(f"Saran tindakan: {scoring['saran']}")

    return summary
