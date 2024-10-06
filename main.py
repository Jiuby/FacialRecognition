import cv2
import face_recognition
import threading
import numpy as np
import os
import queue

img_folder = 'faces'

images = []
names = []
for filename in os.listdir(img_folder):
    img = cv2.imread(os.path.join(img_folder, filename))
    images.append(img)
    names.append(filename.split('.')[0])

known_face_encodings = []
for img in images:
    face_encoding = face_recognition.face_encodings(img)[0]
    known_face_encodings.append(face_encoding)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

frame_queue = queue.Queue()


def capturar_video():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Redimensionar para mejorar la velocidad (opcional)
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        if frame_queue.qsize() < 1:
            frame_queue.put(frame)

        cv2.imshow('Reconocimiento Facial', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def procesar_caras():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            face_locations = face_recognition.face_locations(frame, model="hog")
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                if True in matches:
                    name = names[matches.index(True)]
                    print(f"Bienvenido, {name}")
                else:
                    print("No autorizado")


hilo_video = threading.Thread(target=capturar_video)
hilo_procesamiento = threading.Thread(target=procesar_caras)

# Iniciar ambos hilos
hilo_video.start()
hilo_procesamiento.start()

hilo_video.join()
hilo_procesamiento.join()

cap.release()
cv2.destroyAllWindows()
