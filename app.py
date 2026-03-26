import streamlit as st
from PIL import Image

st.title("👓 Recomendador de Gafas")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Imagen subida", use_column_width=True)
    
    st.success("Imagen cargada correctamente 👌")
    
    st.subheader("Recomendación:")
    st.write("Te recomendamos gafas tipo aviador 😎")
