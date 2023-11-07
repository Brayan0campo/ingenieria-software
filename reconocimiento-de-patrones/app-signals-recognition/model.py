import os
import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ruta de audios
Ruta = "dataset"
Total = 111
Muestreo = 250000
Dataframe = np.zeros((Total, Muestreo + 1))

# Leer las etiquetas
try:
    Labels = pd.read_excel('labels/Labels.xlsx', header=None)
    Y = Labels.iloc[:, 0].values.astype(int)
except Exception as e:
    print(f"Error al cargar etiquetas: {e}")
    exit()

# Extraer características
X = []
for i in range(1, Total + 1):
    for ext in ['mp3', 'aac']:
        Ruta_audios = os.path.join(Ruta, str(i) + '.' + ext)
        if os.path.exists(Ruta_audios):
            try:
                Señal, sr = librosa.load(Ruta_audios)
                Señal_New = np.resize(Señal, Muestreo)

                Magnitud, _ = librosa.magphase(librosa.stft(Señal))
                RMS_vector = librosa.feature.rms(S=Magnitud)
                RMS = RMS_vector.mean()
                Times_vector = librosa.times_like(RMS_vector)
                Times = Times_vector.mean()
                Varianza = np.var(Señal)
                rms_amplitude = np.sqrt(np.mean(np.square(Señal)))
                Desviacion = np.std(Señal)
                zero_crossings = np.where(np.diff(np.sign(Señal)))[0]
                zcr = len(zero_crossings)
                skewness = np.abs(
                    np.mean((Señal - np.mean(Señal)) ** 3) / (np.std(Señal) ** 3))

                """
                fft = np.fft.fft(Señal_New)
                frecuencias = np.fft.fftfreq(len(fft))
                """
                caracteristicas = [RMS, Times, Varianza,
                                   rms_amplitude, Desviacion, zcr, skewness]
                Dataframe[i - 1, 0:Muestreo] = np.transpose(Señal_New)
                Dataframe[i - 1, Muestreo] = Y[i - 1]
                """ Dataframe[i - 1, Muestreo + 1] = max(frecuencias) """
                X.append(caracteristicas)
                break
            except Exception as e:
                print(f"Error en la extracción de características: {e}")
                continue
        else:
            print(f"El archivo de audio {Ruta_audios} no existe.")

# Normalizar características
try:
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
except Exception as e:
    print(f"Error al normalizar características: {e}")

# Dividir el conjunto de datos en entrenamiento y prueba
try:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
except Exception as e:
    print(f"Error al dividir el conjunto de datos: {e}")
    X_train, X_test, y_train, y_test = [], [], [], []

# Entrenar un clasificador KNN
try:
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Error al entrenar el modelo: {e}")

# Realizar predicciones en el conjunto de prueba
try:
    y_pred = model.predict(X_test)
except Exception as e:
    print(f"Error al realizar predicciones: {e}")

# Calcular la precisión del modelo
try:
    accuracy = accuracy_score(y_test, y_pred)
    print(" ")
    print("Precisión del modelo KNN:", accuracy)
    print("-----------------------------------------------------")
except Exception as e:
    print(f"Error al calcular la precisión del modelo: {e}")

# Mostrar reporte de clasificación con precision, recall y F1-score
try:
    report = classification_report(y_test, y_pred)
    print(" ")
    print("Reporte de Clasificación:")
    print(report)
    print("-----------------------------------------------------")
except Exception as e:
    print(f"Error al mostrar el reporte de clasificación: {e}")

# Calcular la matriz de confusión
try:
    confusion = confusion_matrix(y_test, y_pred)
    print(" ")
    print("Matriz de Confusión:")
    print(confusion)
    print(" ")
    print("-----------------------------------------------------")
except Exception as e:
    print(f"Error al calcular la matriz de confusión: {e}")

# Guardado del modelo
try:
    joblib.dump(model, 'signals_model.pkl')
except Exception as e:
    print(f"Error al guardar el modelo: {e}")

# ---------------------------------------------------------------------------------------------- #

# Carga la nueva señal de audio
nueva_senal, sr = librosa.load("dataset/96.mp3")

# Extrae características de la nueva señal
Magnitud, _ = librosa.magphase(librosa.stft(nueva_senal))
RMS_vector = librosa.feature.rms(S=Magnitud)
RMS = RMS_vector.mean()
Times_vector = librosa.times_like(RMS_vector)
Times = Times_vector.mean()
Varianza = np.var(nueva_senal)
rms_amplitude = np.sqrt(np.mean(np.square(nueva_senal)))
Desviacion = np.std(nueva_senal)
zero_crossings = np.where(np.diff(np.sign(nueva_senal)))[0]
zcr = len(zero_crossings)
skewness = np.abs(np.mean((nueva_senal - np.mean(nueva_senal)) ** 3) / (np.std(nueva_senal) ** 3))

# Almacena las características en una lista
caracteristicas_nueva_senal = [RMS, Times, Varianza, rms_amplitude, Desviacion, zcr, skewness]

# Normaliza las características
caracteristicas_nueva_senal_normalizadas = scaler.transform([caracteristicas_nueva_senal])

# Realiza una predicción con el modelo
prediccion = model.predict(caracteristicas_nueva_senal_normalizadas)

# Muestra la etiqueta predicha
print(" ")
print("Etiqueta predicha para la nueva señal:", prediccion)

""""
Precisión del modelo KNN: 1.0
-----------------------------------------------------

Reporte de Clasificación:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        20
           1       1.00      1.00      1.00        14

    accuracy                           1.00        34
   macro avg       1.00      1.00      1.00        34
weighted avg       1.00      1.00      1.00        34

-----------------------------------------------------

Matriz de Confusión:
[[20  0]
 [ 0 14]]

-----------------------------------------------------

Etiqueta predicha para la nueva señal: [0]
"""