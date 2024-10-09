import cv2
import pickle
import numpy as np
from insightface import app
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import threading
import time
import dlib
from scipy.spatial import distance

# Cargar los embeddings y etiquetas previamente guardados con Pickle
with open('embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
    embeddings = data['embeddings']
    labels = data['labels']

# Inicializar el modelo ArcFace más ligero (arcface_mbf para mayor velocidad)
model = app.FaceAnalysis(allowed_modules=['detection', 'recognition'], model_name='arcface_mbf', det_size=(320, 320))  # Reducimos la resolución de detección
model.prepare(ctx_id=0)  # Utiliza GPU si está disponible

# Normalizar los embeddings previamente cargados
embeddings = normalize(embeddings)

# Cargar el detector de dlib para los landmarks faciales
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Asegúrate de tener este archivo

# Índices de los puntos clave de los ojos
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Función para calcular el EAR (Eye Aspect Ratio)
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Umbral para determinar el parpadeo
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 3

# Contador de parpadeos y frames consecutivos
blink_counter = 0
blink_detected = False

# Función para reconocer a una persona en una imagen
def recognize_person(frame, threshold=0.7):
    faces = model.get(frame)

    for face in faces:
        if face.embedding is not None:
            face_embedding = face.embedding.reshape(1, -1)
            face_embedding = normalize(face_embedding)

            similarities = []
            for i, db_embedding in enumerate(embeddings):
                similarity = cosine_similarity(face_embedding, db_embedding.reshape(1, -1))
                similarities.append((labels[i], similarity[0][0]))

            best_match = max(similarities, key=lambda x: x[1])
            if best_match[1] > threshold:
                return best_match[0]
    return "No reconocido"

# Variable para almacenar el resultado del reconocimiento
recognition_result = None

# Función para realizar reconocimiento en segundo plano
def background_recognition(frame):
    global recognition_result
    recognition_result = recognize_person(frame)

# Funcionamiento en tiempo real con OpenCV
cap = cv2.VideoCapture(0)  # Captura desde la webcam

last_recognition_time = 0
recognition_interval = 2  # Intervalo entre reconocimientos

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face_rect in faces:
        shape = predictor(gray, face_rect)
        shape_np = np.zeros((68, 2), dtype="int")

        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)

        left_eye = shape_np[LEFT_EYE]
        right_eye = shape_np[RIGHT_EYE]

        # Calcular EAR para ambos ojos
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Si el EAR cae por debajo del umbral, contamos el parpadeo
        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= EAR_CONSEC_FRAMES:
                print("Parpadeo detectado")
                blink_detected = True
            blink_counter = 0

    # Iniciar reconocimiento si se detecta un parpadeo
    if blink_detected:
        current_time = time.time()
        if current_time - last_recognition_time >= recognition_interval:
            start_time = time.time()

            recognition_thread = threading.Thread(target=background_recognition, args=(frame,))
            recognition_thread.start()
            recognition_thread.join()

            end_time = time.time()
            analysis_time = end_time - start_time
            print(f"Tiempo de análisis: {analysis_time:.2f} segundos")

            if recognition_result is not None:
                cv2.putText(frame, recognition_result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Reconocimiento Facial', frame)
                print(f"Reconocido: {recognition_result}")

            last_recognition_time = current_time
            blink_detected = False

    # Mostrar la imagen en tiempo real sin los párpados marcados
    cv2.imshow('Reconocimiento Facial', frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
