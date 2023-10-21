import cv2
import joblib
import numpy as np
from flask import Flask, request, jsonify
from model import extract_features
from werkzeug.datastructures import FileStorage

app = Flask(__name__)

# Cargar el modelo entrenado
model = joblib.load('emotion_model.pkl')

# Etiquetas de emociones
emotions = ['angry', 'happy', 'neutral', 'sad']

@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        # Obtener la imagen de la solicitud POST
        image_data = request.files.get('image')

        if image_data and isinstance(image_data, FileStorage):

            # El archivo es válido, continúa con el procesamiento
            image = cv2.imdecode(np.fromstring(image_data.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

            # Agregar una declaración de impresión para verificar la forma de la imagen
            print("Forma de la imagen:", image.shape)

            # Extraer características de la imagen
            features = extract_features(image)

            # Agregar declaraciones de impresión para verificar las características extraídas
            print("Características extraídas:", features)

            # Realizar la predicción
            predicted_emotion = model.predict([features])[0]
            emotion_label = emotions.index(predicted_emotion)

            # Agrega una declaración de impresión para verificar la emoción predicha y la etiqueta
            print("Emoción predicha:", predicted_emotion)
            print("Etiqueta de emoción:", emotion_label)

            return jsonify({'emotion': predicted_emotion, 'label': emotion_label})
        else:
            # Manejar el caso en el que no se proporcionó un archivo de imagen válido
            print("No se proporcionó un archivo de imagen válido")
            return jsonify({'error': "No se proporcionó un archivo de imagen válido"})
    except Exception as e:
        print("Error en la API:", str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
