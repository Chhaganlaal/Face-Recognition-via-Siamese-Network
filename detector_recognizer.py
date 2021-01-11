import numpy as np
import model
import cv2
import os
import re
import sys
import tensorflow as tf
import h5py
from utils import triplet_loss, custom_resize, img_to_encoding, draw_on_image, resize_image

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def recognize_video(fr_model, encodings, video_path=0):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(("Error opening camera" if video_path==0 else "Error opening video file"))

    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            faces = face_cascade.detectMultiScale(frame, 1.3, 5)

            for face in faces:
                x, y, w, h = face
                offset = 10

                face_section = frame[y-offset: y+h+offset, x-offset: x+w+offset]
                face_section = custom_resize(face_section, (96, 96))

                prediction = fr_model.recognize(encodings, face_section)

                draw_on_image(frame, prediction, face)

            cv2.imshow("Cam", frame)
            # cv2.imshow("Face", face_section)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
    cv2.destroyAllWindows()

def recognize_image(fr_model, encodings, image_path):
    
    image = cv2.imread(image_path)

    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    for face in faces:
        x, y, w, h = face
        offset = 10

        face_section = image[y-offset: y+h+offset, x-offset: x+w+offset]
        face_section = custom_resize(face_section, (96, 96))

        prediction = fr_model.recognize(encodings, face_section)

        draw_on_image(image, prediction, face)

        print(prediction)

    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Image", resize_image(image).shape[1], resize_image(image).shape[0])
    cv2.imwrite("out.jpg", image)

if __name__=='__main__':
    fr_model = model.fr_model((96, 96, 3), 'fr_model')
    fr_model.load_weights(os.path.join('./weights', 'weights_2000.h5'))
    # fr_model.summary()

    test_img = cv2.imread('./sample.jpg')
    test_img = custom_resize(test_img, (96, 96))

    predicted = None
    best = 1000

    encodings = {}

    with h5py.File('encodings.h5', 'r') as hf:
        for key in hf.keys():
            encodings[key] = np.array(hf.get(key))

    # print("Predicted : ", predicted, "(", best, ")")

    recognize_video(fr_model, encodings)
