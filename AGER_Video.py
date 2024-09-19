import cv2
import numpy as np
from deepface import DeepFace

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

face_pbtxt = "models/opencv_face_detector.pbtxt"
face_pb = "models/opencv_face_detector_uint8.pb"
age_prototxt = "models/age_deploy.prototxt"
age_model = "models/age_net.caffemodel"
gender_prototxt = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"



age_classifications = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_classifications = ['Male', 'Female']

model_mean = (78.4263377603, 87.7689143744, 114.895847746)

flag = True
while True:
    ret, frame = cap.read()
    face_bounds = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        x1, y1, x2, y2 = int(x - 0.18*w), int(y - 0.18*h), int(x + 1.18*w), int(y + 1.18*h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
        face_bounds.append([x1, y1, x2, y2])
    frame_copy = frame.copy()
    cv2.resize(frame_copy, (720, 640))
    
    if(len(faces) == 1):
        if flag:
            current_center = (x2-x1 + 100, y2-y1 + 100)
            flag = False
        
        if abs(current_center[0] - (x2 - x1)) <= ((x2 - x1)+(y2 - y1))/48 and abs(current_center[1] - (y2 - y1)) <= ((x2 - x1)+(y2 - y1))/48:
            for face_bound in face_bounds:
                try:
                    cv2.putText(frame_copy, f'{age}, {gender}, {emotion}, {race}', (face_bound[0], face_bound[1]-10), cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,255),4,cv2.LINE_AA)
                except Exception as e:
                    print(e)
                    continue
        else:
            for face_bound in face_bounds:
                try:
                    result = DeepFace.analyze(frame)
                    age = result[0]['age']
                    gender = result[0]['dominant_gender']
                    emotion = result[0]['dominant_emotion']
                    race = result[0]['dominant_race']
                    cv2.putText(frame_copy, f'{age}, {gender}, {emotion}, {race}', (face_bound[0], face_bound[1]-10), cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,255),4,cv2.LINE_AA)
                except Exception as e:
                    print(e)
                    continue
        current_center = (x2-x1, y2-y1)
    elif (len(faces) > 1):
        flag = True
        for face_bound in face_bounds:
                try:
                    face = frame[face_bound[1]:face_bound[3], face_bound[0]:face_bound[2]]
                    result = DeepFace.analyze(face)
                    age = result[0]['age']
                    gender = result[0]['dominant_gender']
                    emotion = result[0]['dominant_emotion']
                    race = result[0]['dominant_race']
                    cv2.putText(frame_copy, f'{age}, {gender}, {emotion}, {race}', (face_bound[0], face_bound[1]-10), cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,255),4,cv2.LINE_AA)
                except Exception as e:
                    print(e)
                    continue
    cv2.imshow('frame', frame_copy)
    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()
