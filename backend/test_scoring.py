from detection.nihss_scoring import score_nihss

face = "Miring"
hand = "Tangan kanan turun"
voice = "Suara tidak jelas"

result = score_nihss(face, hand, voice)

print("🧠 Total Skor NIHSS:", result['score'])
print("📊 Kategori:", result['kategori'])
print("📌 Rincian:")
for r in result['rincian']:
    print(" -", r)
print("📢 Rekomendasi:", result['saran'])
