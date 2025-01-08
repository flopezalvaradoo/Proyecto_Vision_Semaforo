from typing import List
import numpy as np
import imageio
import cv2
import copy
import glob
import matplotlib.pyplot as plt
import os
import imutils
import pytesseract
from skimage import filters
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_license_plates(image):
    """
    Detecta matrículas en una imagen y devuelve una lista de imágenes recortadas de cada matrícula.

    Args:
        image_path (str): Ruta a la imagen de entrada.

    Returns:
        list: Lista de imágenes recortadas de las matrículas detectadas.
    """

    # Redimensionar para facilitar el procesamiento
    image = cv2.resize(image, (800, 600))

    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro bilateral para reducir el ruido
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)

    # Detectar bordes con Canny
    edges = cv2.Canny(blurred, 30, 200)

    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos para encontrar posibles matrículas
    possible_plates = []
    for contour in contours:
        # Aproximar el contorno
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Buscar formas rectangulares
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 2 < aspect_ratio < 5:  # Relación de aspecto típica de una matrícula
                possible_plates.append((x, y, w, h))

    # Recortar y almacenar las imágenes de las matrículas detectadas
    license_plate_images = []
    for x, y, w, h in possible_plates:
        plate_image = image[y : y + h, x : x + w]
        license_plate_images.append(plate_image)

    return license_plate_images


def validarString(inputString):
    patron = r"^\d{4}[a-zA-Z]{3}$"
    return bool(re.match(patron, inputString))


def extract_license_plate_text(image):
    """
    Detecta el texto de una matrícula en una imagen dada.

    Args:
        image: Imagen de entrada en formato OpenCV.

    Returns:
        str: Texto detectado en la matrícula.
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro bilateral para reducir ruido y mantener bordes
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)

    # Aplicar umbral global (Otsu) para binarizar
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Encontrar contornos en la imagen binarizada
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una máscara para filtrar contornos pequeños
    mask = np.zeros_like(thresh)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filtrar contornos con área pequeña
            cv2.drawContours(mask, [contour], -1, 255, -1)

    # Aplicar la máscara para eliminar ruido
    filtered = cv2.bitwise_and(thresh, thresh, mask=mask)

    # Aplicar OCR con restricciones de caracteres
    text = pytesseract.image_to_string(
        filtered,
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )

    # Limpiar el texto detectado
    clean_text = "".join(filter(str.isalnum, text))

    if len(clean_text) == 8 and validarString(clean_text[1:]):
        return clean_text[1:]

    else:
        return None


def read_video(videopath):
    """
    Reads a video file and returns its frames along with video properties.

    Args:
        videopath (str): The path to the video file.

    Returns:
        tuple: A tuple containing:
            - frames (list): A list of frames read from the video.
            - frame_width (int): The width of the video frames.
            - frame_height (int): The height of the video frames.
            - frame_rate (float): The frame rate of the video.
    """

    cap = cv2.VideoCapture(videopath)

    if not cap.isOpened():
        print("Error: Could not open the video file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, frame_width, frame_height, frame_rate


def obtener_matricula(videopath):
    frames, frame_width, frame_height, frame_rate = read_video(videopath)
    matriculas = {}

    for i, frame in enumerate(frames):
        plates = extract_license_plates(frame)

        for plate in plates:
            codigo = extract_license_plate_text(plate)
            if codigo not in matriculas:
                matriculas[codigo] = 1

            elif codigo is not None:
                matriculas[codigo] += 1

            else:
                pass

    indice = 0
    for matricula in matriculas:
        if matriculas[matricula] > indice:
            indice = matriculas[matricula]

    for matricula in matriculas:
        if matriculas[matricula] == indice:
            matricula_correcta = matricula
    return matricula_correcta


# if __name__ == "__main__":
#     print(obtener_matricula("../data/videos/video_run.avi"))
