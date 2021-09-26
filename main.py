import cv2
from random import randrange

trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# -------------------------------------- for face detection through image --------------------------------------------
# img = cv2.imread('images/face3.jpg')
#
# gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#
# face_coordinates = trained_data.detectMultiScale(gray_img)
# print(face_coordinates)
#
# # for one face only
# # (x, y, w, h) = face_coordinates[0]
# # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
# # # ----------------------------------------------------
#
# # for multi faces
# for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(0, 256), randrange(0, 256), randrange(0, 256)), 2)
# # ----------------------------------------------------
#
#
# cv2.imshow('Face Detection Window', img)
# cv2.waitKey()
# ---------------------------------------------------- Ends Here -----------------------------------------------------

# ----------------------------------------- for face detection through cam --------------------------------------------
# getting video from cam or video file
# webcam = cv2.VideoCapture('video.mp4')
webcam = cv2.VideoCapture(0)

# getting infinite frames until cam is closed or video is ended
while True:
    succesful_frame_read, frame = webcam.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    face_coordinates = trained_data.detectMultiScale(frame)
    print(face_coordinates)

    # for one face only
    # (x, y, w, h) = face_coordinates[0]
    # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # ------------------------------------------------------

    # for multi faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 225), 2)
    # ----------------------------------------------------

    cv2.imshow('Face Detection Window', frame)
    cv2.waitKey(1)
# ---------------------------------------------------- Ends Here -----------------------------------------------------
