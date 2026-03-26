import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

st.title("👓 Recomendador de Gafas")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        st.error("No se detectó rostro")
    else:
        st.image(image, caption="Imagen subida", use_column_width=True)

        st.success("Rostro detectado correctamente 👌")

        # Ejemplo simple de recomendación
        st.subheader("Recomendación:")
        st.write("Te recomendamos gafas tipo aviador 😎")
