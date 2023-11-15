import os
import cv2

# Directorio para imágenes
DATA_DIR = './dataset'

# Crear el directorio si no existe
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Número de clases
classes = 24
classes_size = 100

# Iniciar captura de video
capture = cv2.VideoCapture(0)

for j in range(classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Recopilando datos para la clase {}'.format(j))

    done = False
    while True:
        ret, frame = capture.read()
        cv2.putText(frame, '¿Listo? ¡Presiona "Q"!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0

    # Recopilar imágenes
    while counter < classes_size:
        ret, frame = capture.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Guardar la imagen en el directorio correspondiente
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

capture.release()
cv2.destroyAllWindows()
