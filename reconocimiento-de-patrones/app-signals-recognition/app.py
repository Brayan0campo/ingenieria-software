import joblib
import librosa
from model import scaler
from model import extract_features
from flask import Flask, render_template, request

app = Flask(__name__)

# Cargar el modelo
model = joblib.load('signals_model.pkl')

def predict_audio(audio_file):
    try:
        audio_data, _ = librosa.load(audio_file)
        features = extract_features(audio_data)
        features = scaler.transform([features])
        prediction = model.predict(features)
        return "El motor está en correcto estado" if prediction == 1 else "El motor presenta fallas"
    except Exception as e:
        return str(e)

@app.route('/')
def index():
    return render_template('index.html')

# Ruta para la predicción
@app.route('/predict', methods=['POST'])
def prediction():
    if 'audio' not in request.files:
        return "No se ha seleccionado ningún archivo de audio."
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return "No se ha seleccionado ningún archivo de audio."
    resultado = predict_audio(audio_file)
    return render_template('index.html', resultado=resultado)

if __name__ == '__main__':
    app.run(debug=True)
