import os
import cv2
import pickle
import mediapipe as mp

# Componentes para seguimiento de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Objeto para seguimiento de manos
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './dataset'
images = []
labels = []

try:
    for dir_ in os.listdir(DATA_DIR):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):

            # Listas para almacenar datos y coordenadas
            images_aux = []
            image_x = []
            image_y = []

            # Leer imagen y convertirla a formato RGB
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            if img is None:
                raise Exception(f"No se pudo leer la imagen: {os.path.join(DATA_DIR, dir_, img_path)}")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Procesar imagen para detectar mano
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        image_x.append(x)
                        image_y.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        images_aux.append(x - min(image_x))
                        images_aux.append(y - min(image_y))

                images.append(images_aux)
                labels.append(dir_)

except Exception as e:
    print(f"Error: {str(e)}")

try:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'images': images, 'labels': labels}, f)
    print("Los datos han sido guardados exitosamente")
except Exception as e:
    print(f"Error al guardar los datos: {str(e)}")
