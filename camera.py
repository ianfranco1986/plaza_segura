import time
import threading
import cv2
from face_recognition_knn import entrenamiento, predict_frame, set_label_on_frame
from base_camera import BaseCamera

class Camera(BaseCamera):
    
    @staticmethod
    def set_video_source(source):
         Camera.video_source = source

    @staticmethod
    def frames(self):
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError('No se puede cargar el streaming en la camara: '+str(Camera.camID))
        
        process_this_frame = True
        i = 0 

        while True:
            #print("N° Identificacion: "+str(i))

            # se obtiene el frame 
            _, img = camera.read()

            # se reduce la imagen a 1/4 del tamaño lo que incrementa la velocidad durante el análisis
            small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

            # se transforma la imágen de RBG a RGB 
            rgb_small_frame = small_frame[:, :, ::-1]

            # Evita procesar el mismo frame dos veces 
            if process_this_frame:
                predictions = predict_frame(rgb_small_frame, model_path="trained_knn_model.clf")
                print("Identificación: {} Hilo: {} ".format(predictions, BaseCamera.thread))
                img = set_label_on_frame(img, predictions)



                #for name, (top, right, bottom, left) in predictions:
                #    print("___ Se ha identificado a {} en la posición ({}, {}) ___".format(name, left, top))
                
            process_this_frame = not process_this_frame
            yield cv2.imencode('.jpg', img)[1].tobytes()



#ESTO NO TIENE SENTIDO AQUI
if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Entrenando el clasificador")
    #classifier = train("train", model_save_path="trained_knn_model.clf", n_neighbors=1)
#    print("Entrenamiento Completo!")

    f = Camera()
#    f.run()
