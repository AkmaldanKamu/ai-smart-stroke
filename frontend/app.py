import sys
import os

# Tambahkan path ke folder backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from flask import Flask, render_template, request, jsonify
from detection.voice_detection import detect_speech_clarity

app = Flask(__name__, template_folder="templates", static_folder="static")

# Route utama untuk halaman web
@app.route('/')
def index():
    return render_template("index.html")

# Route API untuk deteksi suara
@app.route('/analyze-audio', methods=['POST'])
def analyze_audio():
    clarity, raw_text = detect_speech_clarity(return_text=True)
    return jsonify({'status': 'ok', 'hasil': clarity, 'transkrip': raw_text})

# Menjalankan server
if __name__ == '__main__':
    app.run(debug=True)
