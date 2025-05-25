import streamlit as st
import face_recognition
import numpy as np
import cv2
from PIL import Image
import os

# Charger les images et leurs noms
path = 'ImagesAttendance'
images = []
classNames = []

for filename in os.listdir(path):
    img = cv2.imread(os.path.join(path, filename))
    images.append(img)
    classNames.append(os.path.splitext(filename)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
    return encodeList

encodeListKnown = findEncodings(images)

# Interface Streamlit
st.title("📸 Système de Reconnaissance Faciale")
uploaded_file = st.file_uploader("📤 Choisissez une photo (comme webcam)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    face_locations = face_recognition.face_locations(img_rgb)
    face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

    for encodeFace, faceLoc in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img_np, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img_np, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            st.success(f"✅ Visage reconnu : {name}")
        else:
            st.warning("❌ Visage inconnu")

    st.image(img_np, channels="RGB", caption="Résultat")

