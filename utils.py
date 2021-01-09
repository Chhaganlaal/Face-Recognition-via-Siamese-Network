import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
import h5py
import matplotlib.pyplot as plt

def triplet_loss(y_true, y_pred, alpha=0.5):

    anchor, positive, negative = y_pred[:, 0, :], y_pred[:, 1, :], y_pred[:, 2, :]
    pos_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1))
    neg_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1))
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss

def conv2d_bn(x, name, layer=None, cv1_out=None, cv1_filter=(1, 1), cv1_strides=(1, 1), cv2_out=None, cv2_filter=(3, 3), cv2_strides=(1, 1), padding=None):

    num = '' if cv2_out == None else '1'
    tensor = Conv2D(cv1_out, cv1_filter, strides=cv1_strides, data_format='channels_last', name=name+'_'+layer+'_conv'+num)(x)
    tensor = BatchNormalization(axis=3, epsilon=0.00001, name=name+'_'+layer+'_bn'+num)(tensor)
    tensor = Activation('relu')(tensor)
    if padding == None:
        return tensor
    tensor = ZeroPadding2D(padding=padding, data_format='channels_last')(tensor)
    if cv2_out == None:
        return tensor
    tensor = Conv2D(cv2_out, cv2_filter, strides=cv2_strides, data_format='channels_last', name=name+'_'+layer+'_conv'+'2')(tensor)
    tensor = BatchNormalization(axis=3, epsilon=0.00001, name=name+'_'+layer+'_bn'+'2')(tensor)
    tensor = Activation('relu')(tensor)
    return tensor

def img_to_encoding(image_path, model):

    img1 = cv2.imread(image_path, 1)
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

def custom_resize(img, inp_dim):
    '''
    Resize without changing aspect ratio
    '''

    img_h, img_w = img.shape[0], img.shape[1]
    w, h = inp_dim
    new_h = int(img_h*min(w/img_w, h/img_h))
    new_w = int(img_w*min(w/img_w, h/img_h))

    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

    return np.uint8(canvas)

def resize_image(img):
    height, width = img.shape[0], img.shape[1]

    resized_image = np.copy(img)

    if max(height, width)>1024:
        ratio = min(1024/height, 1024/width)
        new_h = int(height*ratio)
        new_w = int(width*ratio)

        resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return resized_image

def draw_on_image(image, prediction, face):
    x, y, w, h = face
    offset = 10

    cv2.rectangle(image, (x-offset, y-offset), (x+w+offset, y+h+offset), 2, 1)

    font_scale = max(1, min(image.shape[0], image.shape[1])/(1024))
    t_size = cv2.getTextSize(prediction[0]+": "+str(prediction[1]), cv2.FONT_HERSHEY_PLAIN, font_scale, 1)[0]

    cv2.rectangle(image, (x-offset, y-offset), (x+offset+t_size[0]+3, y+offset+t_size[1]+4), 2, 1)
    cv2.putText(image, prediction[0]+": "+str(prediction[1]), (x+offset, y+offset+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, font_scale, [225,225,225], 1)

def extract(image):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    face_sections = []
    for face in faces:
        x, y, w, h = face
        offset = 10

        face_section = image[y-offset: y+h+offset, x-offset: x+w+offset]

        face_sections.append(face_section)    

    return face_sections

def load_data(dataset_folder, consider_labels=None):
    image_data = {}

    i = 0
    for person in os.listdir(dataset_folder):
        if (consider_labels==None) or (person in consider_labels):
            images = []

            for image_path in os.listdir(os.path.join(dataset_folder, person)):
                image = cv2.imread(os.path.join(dataset_folder, person, image_path))
                image = custom_resize(image, (96, 96))
                images.append(image)

            image_data[person] = np.stack(images)
            i += 1

    return image_data

def get_triplets(image_data):

    label_l, label_r = np.random.choice(list(image_data.keys()), 2, replace=False)
    a, p = np.random.choice(len(image_data[label_l]), 2, replace=False)
    n = np.random.choice(len(image_data[label_r]))

    return image_data[label_l][a], image_data[label_l][p], image_data[label_r][n]

def get_triplets_batch(image_data, n):

    idx_a, idx_p, idx_n = [], [], []
    for _ in range(n):
        a, p, n = get_triplets(image_data)
        idx_a.append(a)
        idx_p.append(p)
        idx_n.append(n)

    return np.stack(idx_a), np.stack(idx_p), np.stack(idx_n)