import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
from PIL import Image

# Function to perform emotion analysis
def analyze_emotion(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    emotions = []
    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']
        emotions.append((x, y, w, h, emotion))

    return emotions

# Streamlit app
st.title('Real-time Emotion Detection')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Run the app
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

while run:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture video")
        break

    # Analyze emotions
    emotions = analyze_emotion(frame)

    # Draw rectangles and labels
    for (x, y, w, h, emotion) in emotions:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()