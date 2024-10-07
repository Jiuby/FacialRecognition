import cv2
import face_recognition
import threading
import numpy as np
import os
import queue

# Carpeta de imágenes
img_folder = 'faces'

# Cargar imágenes y nombres
images = []
names = []
for filename in os.listdir(img_folder):
    img = cv2.imread(os.path.join(img_folder, filename))
    if img is not None:
        images.append(img)
        names.append(filename.split('.')[0])

# Obtener codificaciones faciales conocidas
known_face_encodings = []
for img in images:
    encodings = face_recognition.face_encodings(img)
    if encodings:
        known_face_encodings.append(encodings[0])
    else:
        print(f"No se detectó una cara en la imagen: {filename}")

# Iniciar la captura de video
try:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
except Exception as e:
    print(f"Error al iniciar la cámara: {e}")
    exit()

# Cola para manejar los frames capturados
frame_queue = queue.Queue()

# Evento para detener los hilos
stop_event = threading.Event()

# Contador de frames
frame_count = 0

# Función para capturar video
def capturar_video():
    global frame_count
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        # Redimensionar para mejorar la velocidad
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Procesar solo cada 5 frames
        frame_count += 1
        if frame_count % 5 == 0:
            if frame_queue.qsize() < 1:
                frame_queue.put(frame)

        cv2.imshow('Reconocimiento Facial', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()  # Detener ambos hilos
            break

# Función para procesar las caras
def procesar_caras():
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Localizar caras en el frame
            face_locations = face_recognition.face_locations(frame, model="hog")
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            # Comparar cada cara detectada con las conocidas
            for face_encoding, face_location in zip(face_encodings, face_locations):
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.6:  # Ajusta el umbral según sea necesario
                    name = names[best_match_index]
                    print(f"Bienvenido, {name}")
                else:
                    print("No autorizado")

# Crear e iniciar los hilos
hilo_video = threading.Thread(target=capturar_video)
hilo_procesamiento = threading.Thread(target=procesar_caras)

try:
    hilo_video.start()
    hilo_procesamiento.start()

    hilo_video.join()
    hilo_procesamiento.join()
finally:
    # Liberar los recursos de la cámara y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()