import cv2
import pickle
import numpy as np
from insightface import app
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import threading
import time

# Cargar los embeddings y etiquetas previamente guardados con Pickle
with open('embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
    embeddings = data['embeddings']
    labels = data['labels']

# Inicializar el modelo ArcFace más ligero (arcface_mbf para mayor velocidad)
model = app.FaceAnalysis(allowed_modules=['detection', 'recognition'], model_name='arcface_mbf', det_size=(640, 640))  # Ajustar tamaño de detección
model.prepare(ctx_id=0)  # Utiliza GPU si está disponible

# Normalizar los embeddings previamente cargados
embeddings = normalize(embeddings)

# Función para reconocer una persona en una imagen nueva
def recognize_person(frame, threshold=0.7):
    faces = model.get(frame)

    for face in faces:
        if face.embedding is not None:
            face_embedding = face.embedding.reshape(1, -1)
            face_embedding = normalize(face_embedding)

            # Calcular la similitud entre el embedding de la persona en la imagen y los embeddings guardados
            similarities = []
            for i, db_embedding in enumerate(embeddings):
                similarity = cosine_similarity(face_embedding, db_embedding.reshape(1, -1))
                similarities.append((labels[i], similarity[0][0]))

            # Encontrar la persona con la similitud más alta
            best_match = max(similarities, key=lambda x: x[1])

            if best_match[1] > threshold:
                return best_match[0]  # Retorna el nombre de la persona reconocida

    return "No reconocido"

# Variable para almacenar el resultado del reconocimiento en un hilo separado
recognition_result = None

# Función para realizar reconocimiento en segundo plano
def background_recognition(frame):
    global recognition_result
    recognition_result = recognize_person(frame)

# Funcionamiento en tiempo real con OpenCV
cap = cv2.VideoCapture(0)  # Captura desde la webcam

# Variable para controlar el tiempo entre reconocimientos
last_recognition_time = 0
recognition_interval = 2  # Intervalo de 2 segundos entre reconocimientos

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reducción de la resolución de la imagen
    frame_resized = cv2.resize(frame, (1280, 720))  # Ajustar el tamaño de la imagen para acelerar el análisis

    # Mostrar el frame en la ventana de video
    cv2.imshow('Reconocimiento Facial', frame_resized)

    # Esperar a que el usuario presione la barra espaciadora para reconocer
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Si se presiona la barra espaciadora
        current_time = time.time()
        if current_time - last_recognition_time >= recognition_interval:
            # Medir el tiempo de inicio
            start_time = time.time()

            # Crear un hilo para realizar el reconocimiento en segundo plano
            recognition_thread = threading.Thread(target=background_recognition, args=(frame_resized,))
            recognition_thread.start()

            # Esperar a que el reconocimiento finalice
            recognition_thread.join()

            # Medir el tiempo de finalización
            end_time = time.time()

            # Calcular y mostrar el tiempo de análisis
            analysis_time = end_time - start_time
            print(f"Tiempo de análisis: {analysis_time:.2f} segundos")

            if recognition_result is not None:
                cv2.putText(frame_resized, recognition_result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Reconocimiento Facial', frame_resized)

                # Imprimir el nombre de la persona en la consola
                print(f"Reconocido: {recognition_result}")

            # Actualizar el tiempo del último reconocimiento
            last_recognition_time = current_time

    # Presiona 'q' para salir
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
