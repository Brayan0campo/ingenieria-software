import os
import cv2
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.metrics import accuracy_score, classification_report

# Función para calcular coarseness
def calculate_coarseness(image):
    try:
        image = image.astype('int32')
        m, n = image.shape
        coarseness = np.max(image) - np.min(image)
        return coarseness / (m * n)
    except Exception as e:
        print(f"Error en calculate_coarseness: {str(e)}")
        return None

# Función para calcular contraste de Tamura
def calculate_contrast(image):
    try:
        image = image.astype('int32')
        m, n = image.shape
        mean_intensity = np.mean(image)
        contrast = np.sum(np.square(image - mean_intensity)) / (m * n)
        return contrast
    except Exception as e:
        print(f"Error en calculate_contrast: {str(e)}")
        return None

# Función para calcular directionality de Tamura
def calculate_directionality(image):
    try:
        image = image.astype('int32')
        m, n = image.shape
        grad_x = np.gradient(image, axis=1)
        grad_y = np.gradient(image, axis=0)
        directionality = np.sum(np.abs(grad_x)) + np.sum(np.abs(grad_y))
        return directionality / (m * n)
    except Exception as e:
        print(f"Error en calculate_directionality: {str(e)}")
        return None

# Función para extraer características
def extract_features(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise Exception("No se pudo cargar la imagen correctamente")

        # Calcular la matriz de concurrencia
        glcm = graycomatrix(image, [1], [0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        ASM = graycoprops(glcm, 'ASM')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]

        # Calcular características de Tamura
        coarseness = calculate_coarseness(image)
        contrast_tamura = calculate_contrast(image)
        directionality = calculate_directionality(image)

        """
        # Calcular características LBP
        radius = 1
        n_points = 8 * radius
        lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')
        lbp_histogram, _ = np.histogram(lbp_image, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        """
        # Concatenar todas las características
        all_features = [contrast, dissimilarity, homogeneity, ASM, energy, coarseness, contrast_tamura, directionality]

        """ all_features += list(lbp_histogram) """

        return all_features
    except Exception as e:
        print(f"Error al procesar la imagen {image_path}: {str(e)}")
        return None

# Directorios de datos
train_emotions = 'datasets/train'
test_emotions = 'datasets/test'

# Función para cargar imágenes y etiquetas
def load_images_and_labels(emotion_folder, data):
    for emotion in os.listdir(emotion_folder):
        emotion_path = os.path.join(emotion_folder, emotion)
        for image_name in os.listdir(emotion_path):
            image_path = os.path.join(emotion_path, image_name)
            try:
                image = extract_features(image_path)
                if image is not None:
                    data['X'].append(image)
                    data['y'].append(emotion)
            except Exception as e:
                print(f"Error al cargar la imagen {image_path}: {str(e)}")

# Cargar imágenes y etiquetas de entrenamiento y prueba
data_train = {'X': [], 'y': []}
load_images_and_labels(train_emotions, data_train)

data_test = {'X': [], 'y': []}
load_images_and_labels(test_emotions, data_test)

# Verificar si hay datos cargados
if not data_train['X'] or not data_test['X']:
    print("No se han cargado datos. Revise la estructura de las carpetas y las imágenes")

# Convertir listas de características en matrices
X_train = np.array(data_train['X'])
y_train = np.array(data_train['y'])

X_test = np.array(data_test['X'])
y_test = np.array(data_test['y'])

# Normalización de datos
try:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
except Exception as e:
    print(f"Error al normalizar los datos: {e}")

# Entrenamiento del modelo
try:
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Error al entrenar el modelo: {e}")

# Evaluación del modelo
try:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(" ")
    print("Precisión del modelo en el conjunto de prueba:", accuracy)
    print("-----------------------------------------------------")
    print(classification_report(y_test, y_pred))
except Exception as e:
    print(f"Error al evaluar el modelo: {e}")

# Matriz de confusión
try:
    cm = confusion_matrix(y_test, y_pred)
    print("-----------------------------------------------------")
    print(" ")
    print("Matriz de confusión:")
    print(cm)
    print(" ")
except Exception as e:
    print(f"Error al crear la matriz de confusión: {e}")

# Guardado del modelo
try:
    joblib.dump(model, 'emotion_model.pkl')
except Exception as e:
    print(f"Error al guardar el modelo: {e}")

# ------------------------------------------------------------------------------ #

# Ruta de la imagen a predecir
new_image_path = 'datasets/Sad.jpg'

# Extraer características de la imagen
new_image_features = extract_features(new_image_path)

# Normalizar características
try:
    new_image_features = scaler.transform([new_image_features])
except Exception as e:
    print(f"Error al normalizar los datos de la imagen: {e}")

# Realizar la predicción
try:
    predicted_emotion = model.predict(new_image_features)
    print("Emoción predicha para la imagen:", predicted_emotion)
except Exception as e:
    print(f"Error al predecir la emoción de la imagen: {e}")


"""
Precisión del modelo en el conjunto de prueba: 0.995
-----------------------------------------------------
              precision    recall  f1-score   support

       angry       0.98      1.00      0.99        50
       happy       1.00      0.98      0.99        50
     neutral       1.00      1.00      1.00        50
         sad       1.00      1.00      1.00        50

    accuracy                           0.99       200
   macro avg       1.00      0.99      0.99       200
weighted avg       1.00      0.99      0.99       200

-----------------------------------------------------

Matriz de confusión:
[[50  0  0  0]
 [ 1 49  0  0]
 [ 0  0 50  0]
 [ 0  0  0 50]]

Emoción predicha para la imagen: ['sad']
"""