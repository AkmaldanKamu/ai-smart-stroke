def generate_guidance(score):
    if score == 0:
        return """
        <p><strong>🧘 Tidak ditemukan gejala stroke.</strong></p>
        <ul>
            <li>Pasien dalam kondisi normal, namun tetap perhatikan gejala yang mungkin timbul di kemudian hari.</li>
            <li>Jaga gaya hidup sehat: konsumsi makanan bergizi, olahraga teratur, hindari rokok dan alkohol.</li>
            <li>Jika memiliki riwayat hipertensi, diabetes, atau kolesterol tinggi — lakukan pemeriksaan rutin ke dokter.</li>
        </ul>
        """

    elif score <= 2:
        return """
        <p><strong>⚠️ Gejala ringan terdeteksi.</strong></p>
        <ul>
            <li>Dampingi pasien dan <strong>catat waktu munculnya gejala</strong> pertama kali.</li>
            <li>Jangan berikan makan/minum karena risiko tersedak bila kondisi memburuk.</li>
            <li>Sarankan untuk <strong>periksa ke dokter atau klinik terdekat</strong> guna observasi dan tindak lanjut.</li>
            <li>Pelajari tanda awal stroke seperti senyum tidak simetris, bicara pelo, dan kelemahan pada salah satu sisi tubuh.</li>
        </ul>
        """

    elif score <= 4:
        return """
        <p><strong>🚨 Gejala sedang terdeteksi.</strong></p>
        <ul>
            <li><strong>Bawa segera ke UGD</strong> rumah sakit terdekat, waktu sangat penting (<em>golden hour</em> stroke adalah 3 jam).</li>
            <li>Pastikan pasien dalam posisi <strong>berbaring miring</strong> agar tidak tersedak jika muntah.</li>
            <li>Siapkan dokumen penting: kartu identitas, BPJS/KIS/asuransi, dan daftar obat-obatan atau riwayat medis pasien.</li>
            <li>Berikan edukasi keluarga: stroke bukan hanya kelumpuhan, tapi bisa menyebabkan kesulitan bicara, penglihatan kabur, atau bingung tiba-tiba.</li>
        </ul>
        """

    else:
        return """
        <p><strong>🛑 Gejala berat terdeteksi! Ini adalah <u>darurat medis</u>.</strong></p>
        <ul>
            <li><strong>Segera hubungi 119 atau layanan ambulans</strong>. Jangan tunda!</li>
            <li>Tempatkan pasien dalam posisi menyamping untuk mencegah aspirasi (tersedak/muntah).</li>
            <li>Longgarkan pakaian dan jaga jalan napas tetap terbuka. Jika pasien tidak bernapas, segera lakukan CPR jika terlatih.</li>
            <li><strong>Jangan berikan makanan, minuman, atau obat oral</strong> tanpa seizin medis.</li>
            <li>Berikan edukasi pada keluarga: stroke berat bisa menimbulkan kecacatan permanen jika tidak ditangani cepat. Ajak mereka memahami pentingnya deteksi dini dan rehabilitasi pasca-stroke.</li>
        </ul>
        """