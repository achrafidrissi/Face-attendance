import streamlit as st
import face_recognition
import numpy as np
import cv2
from PIL import Image
import io

st.title("🧑‍🏫 Système de Reconnaissance Faciale")

uploaded_file = st.file_uploader("📤 Choisissez une image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image téléchargée", use_column_width=True)

    img_array = np.array(image)
    face_locations = face_recognition.face_locations(img_array)
    face_encodings = face_recognition.face_encodings(img_array, face_locations)

    st.write(f"🧠 Nombre de visages détectés : {len(face_locations)}")

    for i, (top, right, bottom, left) in enumerate(face_locations):
        cv2.rectangle(img_array, (left, top), (right, bottom), (0, 255, 0), 2)
        st.write(f"👤 Visage {i + 1}")

    st.image(img_array, caption="Résultat", use_column_width=True)
