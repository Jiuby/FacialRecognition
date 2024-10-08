import cv2
import pickle
import numpy as np
import insightface
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Cargar los embeddings y etiquetas previamente guardados con Pickle
with open('embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
    embeddings = data['embeddings']
    labels = data['labels']

# Inicializar el modelo ArcFace
model = insightface.app.FaceAnalysis(allowed_modules=['detection', 'recognition'])
model.prepare(ctx_id=-1)  # Usar GPU si está disponible

# Normalizar los embeddings previamente cargados
embeddings = normalize(embeddings)


# Función para reconocer una persona en una imagen nueva
def recognize_person(frame, threshold=0.6):
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


# Funcionamiento en tiempo real con OpenCV
cap = cv2.VideoCapture(0)  # Captura desde la webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reconocer a la persona en el frame actual
    person_name = recognize_person(frame)

    # Mostrar el resultado en la ventana de video
    cv2.putText(frame, person_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Reconocimiento Facial', frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
