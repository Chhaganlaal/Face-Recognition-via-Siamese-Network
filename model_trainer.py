import numpy as np
import cv2
import os
import re
import sys
import time
import tensorflow as tf
from model import fr_model

from utils import custom_resize, get_triplets_batch, triplet_loss, load_data

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)

if __name__=='__main__':
    image_data = load_data("./dataset")

    save_every = 200
    print_every = 20
    batch_size = 8
    n_iter = 2000
    model_path = './weights/'

    fr_model = fr_model((96, 96, 3), 'fr_model')
    fr_model.define_training_model()

    t_start = time.time()
    for i in range(1, n_iter+1):
        a, p, n = get_triplets_batch(image_data, batch_size)
        loss = fr_model.train_on_batch([a, p, n], [a, p, n])
        # print(loss)
        if i%print_every==0:
            print(f"Time for {i} iterations: {(time.time()-t_start)/60} mins")
            print(f"Train Loss: {loss}")
        # if i%save_every==0:
        #     fr_model.save_weights(os.path.join(model_path, f'weights_{i}.h5'))

