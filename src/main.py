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

from calibration import *
from matric import *
from tracking import *

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

"""
    Este script principal simula el funcionamiento de un sistema de monitoreo de tráfico en un semáforo utilizando visión por computadora.
    
    Descripción del funcionamiento:
    --------------------------------
    1. **Calibración de la cámara:** 
       - La función `calibrar()` devuelve:
         - `rms_calibration`: Error cuadrático medio de la calibración.
         - `intrinsics_calibration`: Matriz intrínseca de la cámara.
         - `extrinsics_calibration`: Parámetros de rotación y traslación.
         - `dist_coeffs_calibration`: Coeficientes de distorsión de la lente.
    
    2. **Estado del semáforo:** 
       - Se define el estado del semáforo (`semaforo`), en este caso, "ROJO".
       - Si el semáforo está en "VERDE", los coches pueden pasar sin restricciones.
       - Si el semáforo está en "ROJO", se analiza el video para verificar si el vehículo cruzó indebidamente.

    3. **Análisis del video de tráfico:**
       - Se define la ruta del video (`videopath`) correspondiente al escenario ("video_stop.avi" o "video_run.avi").
       - Si el semáforo es "ROJO":
         - Se utiliza la función `cruza(videopath)` para verificar si un vehículo ha cruzado la línea de parada.
         - Si `cruza` devuelve `True`, el vehículo es multado.
         - La función `obtener_matricula(videopath)` se utiliza para identificar la matrícula del vehículo.
         - Se imprime un mensaje con el monto de la multa y los puntos perdidos en el carnet.
       - Si `cruza` devuelve `False`, se imprime un mensaje indicando que el vehículo ha respetado la señal roja.

    4. **Ejecución del script:** 
       - La ejecución comienza desde `if __name__ == "__main__":`, asegurando que el script solo se ejecute directamente y no cuando se importa.
    
    Resumen:
    --------
    Este script representa un flujo automatizado para:
    - Calibrar la cámara de monitoreo.
    - Detectar cruces indebidos en semáforo rojo.
    - Identificar la matrícula del vehículo infractor y aplicar una multa.
"""

if __name__ == "__main__":
    (
        rms_calibration,
        intrinsics_calibration,
        extrinsics_calibration,
        dist_coeffs_calibration,
    ) = calibrar()

    # print("Intrinsics:\n", intrinsics_calibration)
    # print("Distortion coefficients:\n", dist_coeffs_calibration)
    # print("Root mean squared reprojection error:\n", rms_calibration)

    # String poner semaforo en verde: "VERDE"
    # String poner semeforo en rojo: "ROJO"
    semaforo = "ROJO"
    print(f"______EL SEMAFORO ESTA EN {semaforo}______")

    # VideoPath cuando para en el Semaforo: "../data/videos/video_stop.avi"
    # VideoPath cuando pasa el Semaforo: "../data/videos/video_run.avi"
    videopath = "../data/videos/video_run.avi"

    if semaforo == "ROJO":
        multa = cruza(videopath)
        if multa:
            print("El vehiculo va ser multado")
            print("Se procede a analizar el video para obtener la matricula . . .")
            matricula = obtener_matricula(videopath)
            print(
                f"El vehiculo con matricula: {matricula} es multado con 600$ y 4 puntos en el carnet."
            )

        else:
            print("El coche ha parado correctamente")

    else:
        print("Los coches pueden pasar.")
