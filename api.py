from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from PIL import Image

app = Flask(__name__)

CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})

try:
    model = joblib.load('model_multiclass.pkl')
    print("Model klasifikasi berhasil dimuat.")
except FileNotFoundError:
    print("CRITICAL ERROR: File 'model_multiclass.pkl' tidak ditemukan!")
    print("Pastikan Anda sudah melatih modelnya terlebih dahulu.")
    model = None

def extract_features(image_file):
    with Image.open(image_file) as img:
        img_array = np.array(img)
        avg_rgb = np.mean(img_array, axis=(0, 1))
        return [avg_rgb[0], avg_rgb[1], avg_rgb[2]]

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Server model tidak siap"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "File belum dipilih"}), 400
    
    try:
        fitur_gambar = extract_features(file)

        prediksi = model.predict([fitur_gambar])
        
        nama_tanaman = prediksi[0]
        
        return jsonify({
            "prediksi": nama_tanaman
        })
        
    except Exception as e:
        return jsonify({"error": f"Gagal memproses gambar: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)