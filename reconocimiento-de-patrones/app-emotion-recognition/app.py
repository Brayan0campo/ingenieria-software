import os
import requests
import subprocess
from flask import Flask, request, render_template
from werkzeug.datastructures import FileStorage

app = Flask(__name__)

# Ruta para la página de inicio
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para el formulario de carga de imágenes
@app.route('/upload', methods=['POST'])
def upload_image():
    uploaded_file = request.files.get('image')

    if uploaded_file and isinstance(uploaded_file, FileStorage):
        image_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(image_path)

        # Enviar la imagen a la API
        api_url = 'http://localhost:5000/predict'
        files = {'image': open(image_path, 'rb')}
        response = requests.post(api_url, files=files)
        emotion_data = response.json()

        # Agrega una declaración de impresión para verificar la respuesta de la API
        print("Respuesta de la API:", emotion_data)

        emotion = emotion_data.get('emotion', 'Emoción no encontrada')
        label = emotion_data.get('label', -1)

        return render_template('result.html', emotion=emotion, label=label)
    else:
        # Manejar el caso en el que 'image' no se proporcionó o no es una instancia de 'FileStorage'
        print("No se proporcionó un archivo de imagen válido")
        return render_template('index.html')

if __name__ == '__main__':
    # Iniciar la aplicación web
    app.run(debug=True, port=5001)
