import os
import cv2
import pickle
import numpy as np
import insightface
from sklearn.preprocessing import normalize
from tqdm import tqdm
import time

# Inicializar el modelo ArcFace (usa CPU con ctx_id=-1 o GPU con ctx_id=0)
model = insightface.app.FaceAnalysis(allowed_modules=['detection', 'recognition'])
model.prepare(ctx_id=-1)


# Función para cargar imágenes y extraer embeddings con barra de progreso
def load_images_and_extract_embeddings(image_folder):
    embeddings = []
    labels = []
    image_files = os.listdir(image_folder)

    # Usar tqdm para mostrar barra de progreso
    start_time = time.time()
    for file_name in tqdm(image_files, desc="Extrayendo embeddings", unit="imagen"):
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
                labels.append(file_name.split('.')[0])  # El nombre del archivo sin extensión como etiqueta

    # Mostrar el tiempo total tomado
    elapsed_time = time.time() - start_time
    print(f"Proceso completado en {elapsed_time:.2f} segundos")

    return np.array(embeddings), labels


# Función para normalizar los embeddings
def normalize_embeddings(embeddings):
    return normalize(embeddings)


# Directorio de imágenes
image_folder = 'faces'  # Cambia esto a la carpeta donde están tus imágenes

# Cargar y extraer embeddings
embeddings, labels = load_images_and_extract_embeddings(image_folder)

# Normalizar embeddings
embeddings = normalize_embeddings(embeddings)

# Guardar embeddings y etiquetas en un archivo usando Pickle
with open('embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'labels': labels}, f)

print("Embeddings y etiquetas guardados en 'embeddings.pkl'")
