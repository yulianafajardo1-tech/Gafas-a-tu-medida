import os
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

# Descargar automáticamente si no existe
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
import streamlit as st
import cv2
import math
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.title("👓 Recomendador de Gafas con IA")

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file:

    # Convertir imagen
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Modelo (IMPORTANTE: debes subir este archivo a GitHub también)
    MODEL_PATH = "face_landmarker.task"

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1
    )

    def distancia(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def a_pixeles(lm, w, h):
        return (int(lm.x * w), int(lm.y * h))

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        st.error("No se detectó rostro")
    else:
        lm = result.face_landmarks[0]

        frente = a_pixeles(lm[10], w, h)
        barbilla = a_pixeles(lm[152], w, h)
        lado_izq = a_pixeles(lm[234], w, h)
        lado_der = a_pixeles(lm[454], w, h)

        alto_rostro = distancia(frente, barbilla)
        ancho_rostro = distancia(lado_izq, lado_der)

        ratio = alto_rostro / ancho_rostro

        # Clasificación simple
        if ratio > 1.5:
            tipo = "Rostro alargado"
            recomendacion = "Gafas grandes o rectangulares"
        else:
            tipo = "Rostro más equilibrado"
            recomendacion = "Gafas tipo aviador"

        # Mostrar resultados
        st.image(image, caption="Imagen subida", use_column_width=True)
        st.subheader(f"Tipo de rostro: {tipo}")
        st.success(f"Recomendación: {recomendacion}")
