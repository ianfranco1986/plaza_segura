import time
import threading
import logging

try:
    from greenlet import getcurrent as get_ident
except ImportError:
    try:
        from thread import get_ident
    except ImportError:
        from _thread import get_ident

# Objetivo de la clase:
# Avisar a todos los clientes activos de que nuevos frames se encuentran disponibles
# Código forma parte del servidor Flask

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

class CameraEvent(object):

    def __init__(self):
        self.events = {}

    def wait(self):
        """Invoked from each client's thread to wait for the next frame."""
        ident = get_ident()
        if ident not in self.events:
            # Este es nuevo cliente
            # se agrega una entrada para el "events"
            # cada entrada tiene 2 elementos, un thread.Event() y un timestamp
            self.events[ident] = [threading.Event(), time.time()]
        return self.events[ident][0].wait()

    def set(self):
        """Invoked by the camera thread when a new frame is available."""
        now = time.time()
        remove = None
        for ident, event in self.events.items():
            if not event[0].isSet():
                # if this client's event is not set, then set it
                # also update the last set timestamp to now
                event[0].set()
                event[1] = now
            else:
                # if the client's event is already set, it means the client
                # did not process a previous frame
                # if the event stays set for more than 5 seconds, then assume
                # the client is gone and remove it
                if now - event[1] > 5:
                    remove = ident
        if remove:
            del self.events[remove]

    def clear(self):
        """Invoked from each client's thread after a frame was processed."""
        self.events[get_ident()][0].clear()


class BaseCamera(object):
    thread = None  # Hilo que lee los frames recibidos de la cámara 
    frame = None  # Almacenamiento del frame actual por el hilo activo 
    last_access = 0  # Tiempo que el lleva ejecutado en el lado del cliente
    event = CameraEvent()

    def __init__(self):
        """Start the background camera thread if it isn't running yet."""
        
        if BaseCamera.thread is None:
            BaseCamera.last_access = time.time()

            # Inicio del hilo
            BaseCamera.thread = threading.Thread(target=self._thread)
            BaseCamera.thread.start()


            # Espera hasta recibir un frame
            while self.get_frame() is None:
                time.sleep(0)

    def get_frame(self):
        """Return the current camera frame."""
        BaseCamera.last_access = time.time()

        # wait for a signal from the camera thread
        BaseCamera.event.wait()
        BaseCamera.event.clear()

        return BaseCamera.frame

    @staticmethod
    def get_video_source(self):
        raise RuntimeError('Debe ser implementado por subclases')


    @staticmethod
    def set_source(self, cam):
        raise RuntimeError('Debe ser implementado por la subclase')

    @staticmethod
    def frames(self):
        """"Generator that returns frames from the camera."""
        raise RuntimeError('Debe ser implementado por subclases')

    @classmethod
    def _thread(cls):
        """Camera background thread."""
        print('Iniciando hilo asociado a la cámara')
        frames_iterator = cls.frames(cls)
        for frame in frames_iterator:
            BaseCamera.frame = frame
            BaseCamera.event.set()  # Notifica a los clientes activos 
            time.sleep(0)

            # Si no existen clientes activos detiene la cámara después de 10 segundos 
            if time.time() - BaseCamera.last_access > 20:
                frames_iterator.close()
                print('Deteniendo el hilo')
                break
        BaseCamera.thread = None