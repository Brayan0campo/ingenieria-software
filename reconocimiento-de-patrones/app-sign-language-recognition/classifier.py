import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

try:
    # Cargar el conjunto de datos
    data_dict = pickle.load(open('./data.pickle', 'rb'))

    # Extraer imagenes y etiquetas
    x = np.asarray(data_dict['images'])
    y = np.asarray(data_dict['labels'])

    # Dividir datos en entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)

    # Entrenar clasificador(KNN)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train, y_train)

    # Calcular precisión del modelo
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    print('{}% de las muestras fueron clasificadas correctamente.'.format(score * 100))

    # Guardar modelo entrenado
    with open('model.p', 'wb') as f:
        pickle.dump({'model': model}, f)

except FileNotFoundError:
    print("Error: Archivo no encontrado.")

except Exception as e:
    print(f"Error: {e}")
