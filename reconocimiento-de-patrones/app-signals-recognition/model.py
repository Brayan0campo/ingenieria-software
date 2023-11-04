import os
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
Labels = pd.read_excel('labels/Labels.xlsx', header=None)
Y = Labels.iloc[:, 0].values.astype(int)

X = []
for i in range(1, Total + 1):
    for ext in ['mp3', 'aac']:
        Ruta_audios = os.path.join(Ruta, str(i) + '.' + ext)
        if os.path.exists(Ruta_audios):
            Señal, sr = librosa.load(Ruta_audios)
            Señal_New = np.resize(Señal, Muestreo)

            # Extraer características
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

            skewness = np.abs(np.mean((Señal - np.mean(Señal)) ** 3) / (np.std(Señal) ** 3))

            """ fft = np.fft.fft(Señal_New)
            frecuencias = np.fft.fftfreq(len(fft)) """

            # Almacenar vector de características
            caracteristicas = [RMS, Times, Varianza, rms_amplitude, Desviacion, zcr, skewness]
            Dataframe[i - 1, 0:Muestreo] = np.transpose(Señal_New)
            Dataframe[i - 1, Muestreo] = Y[i - 1]
            """ Dataframe[i - 1, Muestreo + 1] = max(frecuencias) """
            X.append(caracteristicas)
            break

# Normalizar características
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Entrenar un clasificador KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = knn.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(" ")
print("Precisión del modelo KNN:", accuracy)
print("-----------------------------------------------------")

# Mostrar reporte de clasificación con precision, recall y F1-score
report = classification_report(y_test, y_pred)
print(" ")
print("Reporte de Clasificación:")
print(report)
print("-----------------------------------------------------")

# Calcular la matriz de confusión
confusion = confusion_matrix(y_test, y_pred)
print(" ")
print("Matriz de Confusión:")
print(confusion)



# Ruta del nuevo audio que deseas predecir
nuevo_audio_path = "dataset/80.mp3"  # Reemplaza con la ruta del nuevo audio

if os.path.exists(nuevo_audio_path):
    # Cargar el nuevo audio
    Señal_nuevo, sr_nuevo = librosa.load(nuevo_audio_path)
    Señal_nuevo = np.resize(Señal_nuevo, Muestreo)  # Redimensionar a la longitud de Muestreo

    # Extraer características del nuevo audio
    Magnitud_nuevo, _ = librosa.magphase(librosa.stft(Señal_nuevo))
    RMS_vector_nuevo = librosa.feature.rms(S=Magnitud_nuevo)
    RMS_nuevo = RMS_vector_nuevo.mean()
    Times_vector_nuevo = librosa.times_like(RMS_vector_nuevo)
    Times_nuevo = Times_vector_nuevo.mean()
    Varianza_nuevo = np.var(Señal_nuevo)
    rms_amplitude_nuevo = np.sqrt(np.mean(np.square(Señal_nuevo)))
    Desviacion_nuevo = np.std(Señal_nuevo)
    zero_crossings_nuevo = np.where(np.diff(np.sign(Señal_nuevo)))[0]
    zcr_nuevo = len(zero_crossings_nuevo)
    skewness_nuevo = np.abs(np.mean((Señal_nuevo - np.mean(Señal_nuevo)) ** 3) / (np.std(Señal_nuevo) ** 3))

    # Normalizar las características del nuevo audio
    caracteristicas_nuevo = [RMS_nuevo, Times_nuevo, Varianza_nuevo, rms_amplitude_nuevo, Desviacion_nuevo, zcr_nuevo, skewness_nuevo]
    caracteristicas_nuevo = scaler.transform([caracteristicas_nuevo])

    # Realizar la predicción en el nuevo audio
    prediccion = knn.predict(caracteristicas_nuevo)

    # Imprimir el resultado
    if prediccion[0] == 0:
        print(" ")
        print("El coche está malo (predicción: 0)")
    else:
        print(" ")
        print("El coche está bueno (predicción: 1)")
else:
    print("El archivo de audio no se encuentra en la ruta especificada.")
