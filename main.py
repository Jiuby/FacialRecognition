import os
import cv2
import numpy as np
import insightface
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Inicializa el modelo ArcFace (usa CPU con ctx_id=-1)
model = insightface.app.FaceAnalysis(allowed_modules=['detection', 'recognition'])
model.prepare(ctx_id=-1)  # Cambia a un id diferente si tienes GPU (por ejemplo, ctx_id=0)


# Función para cargar imágenes desde una carpeta y extraer embeddings
def load_images_and_extract_embeddings(image_folder):
    embeddings = []
    labels = []

    for file_name in os.listdir(image_folder):
        # Construir ruta completa de la imagen
        img_path = os.path.join(image_folder, file_name)

        # Leer la imagen
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Detectar y extraer embeddings faciales
        faces = model.get(img)
        for face in faces:
            if face.embedding is not None:
                embeddings.append(face.embedding)
                labels.append(
                    file_name.split('.')[0])  # El nombre de la persona es el nombre del archivo sin la extensión

    return np.array(embeddings), labels


# Función para normalizar los embeddings
def normalize_embeddings(embeddings):
    return normalize(embeddings)


# Cargar y extraer embeddings de las imágenes
image_folder = 'data'  # Cambia a la ruta donde están tus imágenes
embeddings, labels = load_images_and_extract_embeddings(image_folder)

# Normalizar embeddings
embeddings = normalize_embeddings(embeddings)

# Guardar embeddings y etiquetas para el reconocimiento futuro
database = {label: emb for label, emb in zip(labels, embeddings)}


# Función para reconocer una persona en una imagen nueva
def recognize_person(frame, threshold=0.6):
    faces = model.get(frame)

    for face in faces:
        if face.embedding is not None:
            face_embedding = face.embedding.reshape(1, -1)
            face_embedding = normalize(face_embedding)

            # Calcula la similitud entre el embedding de la persona en la imagen y los embeddings de la base de datos
            similarities = []
            for name, db_embedding in database.items():
                similarity = cosine_similarity(face_embedding, db_embedding.reshape(1, -1))
                similarities.append((name, similarity[0][0]))

            # Encuentra la persona con la similitud más alta
            best_match = max(similarities, key=lambda x: x[1])

            if best_match[1] > threshold:
                print(f"Bienvenido {best_match[0]}")  # Mostrar en consola
                return best_match[0]  # Retorna el nombre de la persona reconocida

    return "NO RECONOCIDO"


# Funcionamiento en tiempo real con OpenCV
cap = cv2.VideoCapture(0)  # Captura desde la webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reconoce a la persona en el frame actual
    person_name = recognize_person(frame)

    # Configuración del texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    if person_name != "NO RECONOCIDO":
        text = "Bienvenido"
        color = (0, 255, 0)  # Verde para reconocido
    else:
        text = "NO RECONOCIDO"
        color = (0, 0, 255)  # Rojo para no reconocido

    # Mostrar el texto en la ventana de video con una fuente moderna y colores que combinen
    cv2.putText(frame, text, (50, 50), font, 1.5, color, 3, cv2.LINE_AA)

    # Muestra el frame en la ventana
    cv2.imshow('Reconocimiento Facial', frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
