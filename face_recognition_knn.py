import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
import cv2
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def entrenamiento(folder):
    # arrays que contendrán los datos de las coordenadas del rostro identificado y la etiqueda de la persona
    coordenadas = []
    etiqueta = []

    # recorre todas las imágenes dentro del directorio de cada persona
    for persona_dir in os.listdir(folder):
        if not os.path.isdir(os.path.join(folder, persona_dir)):
            continue
        for path in image_files_in_folder(os.path.join(folder, persona_dir)):
            # carga la imagen
            imagen = face_recognition.load_image_file(path)
            # busca rostros en la imagen
            face_box = face_recognition.face_locations(imagen)

            # si existe un rostro detectado, almacena las coordenadas y la etiqueta de la persona 
            if len(face_box) == 1:
                coordenadas.append(face_recognition.face_encodings(imagen, known_face_locations=face_box))
                etiqueta.append(persona_dir)
            elif len(face_box) > 1:
                print("Hay demasiados rostros en la imagen")
            else:
                print("No se han encontrado rostros en la imagen")

    # obtiene automáticamente el n° de vecinos         
    n_neighbors = int(round(math.sqrt(len(coordenadas))))

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    
    knn_clf.fit(coordenadas, etiqueta)

    # guardado del modelo 
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
            print(knn_clf)

    return knn_clf

def predict_frame(X_img, knn_clf=None, model_path=None, distance_threshold=0.5):
    
    if knn_clf is None and model_path is None:
        raise Exception("No se encuentra el modelo ")

    # carga el modelo
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # recibe un frame como parámetro y devuelve una lista de tuplas con las rostros detectados
    X_face_locations = face_recognition.face_locations(X_img)

    # si no encuentra rostros en el frame, devuelve un array vacío 
    if len(X_face_locations) == 0:
        return []

    # Devuelve los rostros codificadas si es que encontró 
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Aplica el algoritmo KNN para encontrar las menores distancias entre los rostros codificados y el modelo previamente cargado
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Devuelve las personas identificadas
    return [(pred, loc) if rec else ("desconocido", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def set_label_on_frame(frame, predictions):
    
    for name, (top, right, bottom, left) in predictions:
    
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Dibuja un rectángulo en el rostro
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)
        
        # Agrega una etiqueta con el nombre
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)    
    
        # retorna el frame ya procesado, el cual es utilizado por la clase camera
    return frame


if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Training KNN classifier...")
    classifier = train("train", model_save_path="trained_knn_model.clf", n_neighbors=1)
    print("Training complete!")

    # STEP 2: Using the trained classifier, make predictions for unknown images
    for image_file in os.listdir("test"):
        full_file_path = os.path.join("test", image_file)

        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        # Display results overlaid on an image
        show_prediction_labels_on_image(os.path.join("test", image_file), predictions)
