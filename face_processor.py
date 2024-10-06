import face_recognition
import cv2
import pickle
import os

# Carpeta con imágenes de entrenamiento
img_folder = 'faces'

# Cargar imágenes de entrenamiento
images = []
names = []
for filename in os.listdir(img_folder):
    img_bgr = cv2.imread(os.path.join(img_folder, filename))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    images.append(img_rgb)
    names.append(filename.split('.')[0])  # Asignar nombre al estudiante

# Codificar imágenes de entrenamiento
known_face_encodings = face_recognition.face_encodings(images)

# Guardar datos en un archivo
with open('face_data.pkl', 'wb') as f:
    pickle.dump((known_face_encodings, names), f)