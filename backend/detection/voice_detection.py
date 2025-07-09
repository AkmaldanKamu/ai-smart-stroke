# backend/detection/voice_detection.py

import whisper
import os
import tempfile
import time
import sounddevice as sd
import scipy.io.wavfile

# Load model Whisper hanya sekali
model = whisper.load_model("base")  # bisa diganti 'base' kalau ingin lebih ringan

def record_audio(duration=5, sample_rate=16000, device_index=2):
    """
    Merekam suara dari microphone dan menyimpannya sebagai file .wav.
    """
    print("▶️ Merekam suara... Silakan bicara.")
    try:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, device=device_index)
        sd.wait()

        volume_level = float(audio.max())
        print(f"🎚️ Volume Max (debug): {volume_level:.4f}")
        if volume_level < 0.01:
            print("⚠️ Suara terlalu kecil. Pastikan mic aktif dan dekat.")

        # Simpan file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            scipy.io.wavfile.write(f.name, sample_rate, audio)
            print(f"🎧 Audio disimpan di: {f.name}")
            return f.name
    except Exception as e:
        print(f"❌ ERROR saat merekam: {e}")
        return None

def detect_speech_clarity(return_text=False):
    """
    Merekam suara, transkrip menggunakan Whisper, dan menentukan apakah suara jelas.
    Jika return_text=True, akan mengembalikan juga teks mentah hasil transkrip.
    """
    audio_file = record_audio()
    if audio_file is None:
        return ("Gagal merekam suara", "") if return_text else "Gagal merekam suara"

    print("🔍 Menganalisis suara...")
    try:
        result = model.transcribe(audio_file, language='id')
        text = result.get("text", "").strip()
        print(f"🗣️ Transkrip: {text}")

        os.remove(audio_file)

        if not text:
            return ("Tidak ada suara", "") if return_text else "Tidak ada suara"

        if "makan" in text.lower() and "nasi" in text.lower():
            return ("Suara jelas", text) if return_text else "Suara jelas"
        else:
            return ("Suara tidak jelas", text) if return_text else "Suara tidak jelas"

    except Exception as e:
        print(f"❌ Error Whisper: {e}")
        return (f"Error analisis suara: {e}", "") if return_text else f"Error analisis suara: {e}"
