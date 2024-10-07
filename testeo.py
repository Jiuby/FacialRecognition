import cv2
import face_recognition
import pickle
import threading
import queue
import numpy as np

# Cargar codificaciones faciales previamente guardadas
with open('face_encodings_with_names.pkl', 'rb') as f:
    data = pickle.load(f)
known_face_encodings = data["encodings"]
names = data["names"]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame_queue = queue.Queue()
stop_event = threading.Event()

# Función para capturar video
def capturar_video():
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        # Redimensionar para mejorar la velocidad
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

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
