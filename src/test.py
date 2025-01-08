import cv2
from picamera2 import Picamera2

def stream_video():
    picam = Picamera2()
    picam.preview_configuration.main.size=(320, 180)
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    while True:
        frame = picam.capture_array()
        cv2.imshow("picam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def take_picture():
    picam = Picamera2()
    picam.preview_configuration.main.size=(320, 180)
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    picam.start_and_capture_file("calibration/picture3.jpg")


def film_video():
    # Inicializa la cámara
    picam = Picamera2()
    picam.preview_configuration.main.size = (672, 378)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    # Configuración para guardar el video
    output_filename = "video_10.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec para el archivo AVI
    fps = 18 # Fotogramas por segundo
    frame_size = (672, 378)  # Tamaño del video
    out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

    try:
        while True:
            frame = picam.capture_array()
            cv2.imshow("picam", frame)

            # Graba el frame actual en el archivo de salida
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Libera los recursos
        picam.stop()
        out.release()
        cv2.destroyAllWindows()


def play_video(file_path):
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("Error: No se pudo abrir el archivo de video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:  # Salir si no hay más frames
            break

        cv2.imshow("Reproduciendo video", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):  # Presiona 'q' para salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # play_video("video_4.avi")
    # stream_video()
    film_video()

