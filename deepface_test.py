import cv2

from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread("/Users/riteshmanchikanti/Work/DEVFI/facial_image.jpeg")
while True:
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print("entering into for")
    for (x, y, w, h) in faces:
        face_roi = rgb_frame[y:y + h, x:x + w]
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        print("Result is - ",result)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('Real-time Emotion Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# rgb_frame = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
# faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
# for (x, y, w, h) in faces:
#         # Extract the face ROI (Region of Interest)
#         face_roi = rgb_frame[y:y + h, x:x + w]

        
#         # Perform emotion analysis on the face ROI
#         result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

#         # Determine the dominant emotion
#         emotion = result[0]['dominant_emotion']
#         # print("The emotion is - ",emotion)
#         print("Result is - ",result)

#         # Draw rectangle around face and label with predicted emotion
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#         cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
# cv2.imshow('Real-time Emotion Detection', img)