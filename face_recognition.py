#!/usr/bin/env python3

import os
import cv2

def face_recognition():
    face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    eye_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    cap_cam = cv2.VideoCapture(0)
    if cap_cam.isOpened():
        while True:
            ret, frame = cap_cam.read()
            if ret is True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                if 0 < len(faces):
                    for face_x, face_y, face_w, face_h in faces:
                        face_pt1 = (face_x, face_y)
                        face_pt2 = (face_x+face_w, face_y+face_h)
                        cv2.rectangle(frame, face_pt1, face_pt2, (255, 0, 0), thickness=2)

                        face = gray[face_y:face_y+face_h, face_x:face_x+face_w]

                        eyes = eye_cascade.detectMultiScale(face, scaleFactor=1.3, minNeighbors=5)
                        if 0 < len(eyes):
                            for eye_x, eye_y, eye_w, eye_h in eyes:
                                eye_pt1 = (face_x+eye_x, face_y+eye_y)
                                eye_pt2 = (face_x+eye_x+eye_w, face_y+eye_y+eye_h)
                                cv2.rectangle(frame, eye_pt1, eye_pt2, (0, 255, 0), thickness=2)

                cv2.imshow('Capture', frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    break
    cap_cam.release()

if __name__ == '__main__':
    face_recognition()
