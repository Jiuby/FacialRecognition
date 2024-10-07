import face_recognition
import os
import pickle

# Carpeta de imágenes
img_folder = 'faces'

# Listas para almacenar las imágenes y nombres
images = []
names = []

for filename in os.listdir(img_folder):
    img = face_recognition.load_image_file(os.path.join(img_folder, filename))
    images.append(img)
    names.append(filename.split('.')[0])  # Usamos el nombre del archivo como nombre de la persona

known_face_encodings = []

for img, name in zip(images, names):
    encodings = face_recognition.face_encodings(img)
    if encodings:
        known_face_encodings.append(encodings[0])  # Guardar la primera codificación

# Guardar las codificaciones y nombres en un archivo pickle
data = {"encodings": known_face_encodings, "names": names}
with open('face_encodings_with_names.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Codificaciones y nombres guardados en 'face_encodings_with_names.pkl'")
