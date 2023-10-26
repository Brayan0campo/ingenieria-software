import cv2
import joblib
import numpy as np
from model import scaler
from model import extract_features
from flask import Flask, request, render_template

app = Flask(__name__)

# Cargar el modelo previamente entrenado
model = joblib.load('emotion_model.pkl')

# Función para predecir la emoción
def predict_emotion(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        features = extract_features(image_path)
        if features is not None:
            features = np.array(features).reshape(1, -1)
            features = scaler.transform(features)
            emotion = model.predict(features)[0]
            return emotion
        else:
            return "No se pudo procesar la imagen correctamente"
    except Exception as e:
        return str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No se ha subido ninguna imagen."

    image = request.files['image']
    if image.filename == '':
        return "No se ha seleccionado ninguna imagen."

    image_path = 'temp_image.jpg'
    image.save(image_path)
    emotion = predict_emotion(image_path)

    return render_template('result.html', emotion=emotion)

if __name__ == '__main__':
    app.run(debug=True)
