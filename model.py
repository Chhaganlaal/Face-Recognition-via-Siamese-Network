import numpy as np
import cv2
import utils
import os
import re
import sys
import time
import tensorflow as tf

from utils import triplet_loss

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, AveragePooling2D, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, Input, Lambda, Layer, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.models import Model

from tensorflow.keras.initializers import glorot_uniform

from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


class fr_model:

    def __init__(self, input_shape, name):
        self.input_shape = input_shape
        self.model = self.inception_model(Input(input_shape), name)

    def inception_block_1a(self, X, name):
        """
        Implementation of an inception block
        """
        
        X_3x3 = Conv2D(96, (1, 1), data_format='channels_last', name=name+'_inception_3a_3x3_conv1')(X)
        X_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name=name+'_inception_3a_3x3_bn1')(X_3x3)
        X_3x3 = Activation('relu')(X_3x3)
        X_3x3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(X_3x3)
        X_3x3 = Conv2D(128, (3, 3), data_format='channels_last', name=name+'_inception_3a_3x3_conv2')(X_3x3)
        X_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name=name+'_inception_3a_3x3_bn2')(X_3x3)
        X_3x3 = Activation('relu')(X_3x3)
        
        X_5x5 = Conv2D(16, (1, 1), data_format='channels_last', name=name+'_inception_3a_5x5_conv1')(X)
        X_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name=name+'_inception_3a_5x5_bn1')(X_5x5)
        X_5x5 = Activation('relu')(X_5x5)
        X_5x5 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(X_5x5)
        X_5x5 = Conv2D(32, (5, 5), data_format='channels_last', name=name+'_inception_3a_5x5_conv2')(X_5x5)
        X_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name=name+'_inception_3a_5x5_bn2')(X_5x5)
        X_5x5 = Activation('relu')(X_5x5)

        X_pool = MaxPooling2D(pool_size=3, strides=2, data_format='channels_last')(X)
        X_pool = Conv2D(32, (1, 1), data_format='channels_last', name=name+'_inception_3a_pool_conv')(X_pool)
        X_pool = BatchNormalization(axis=3, epsilon=0.00001, name=name+'_inception_3a_pool_bn')(X_pool)
        X_pool = Activation('relu')(X_pool)
        X_pool = ZeroPadding2D(padding=((3, 4), (3, 4)), data_format='channels_last')(X_pool)

        X_1x1 = Conv2D(64, (1, 1), data_format='channels_last', name=name+'_inception_3a_1x1_conv')(X)
        X_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name=name+'_inception_3a_1x1_bn')(X_1x1)
        X_1x1 = Activation('relu')(X_1x1)
            
        # CONCAT
        inception = Concatenate(3)([X_3x3, X_5x5, X_pool, X_1x1])

        return inception

    def inception_block_1b(self, X, name):
        X_3x3 = Conv2D(96, (1, 1), data_format='channels_last', name=name+'_inception_3b_3x3_conv1')(X)
        X_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name=name+'_inception_3b_3x3_bn1')(X_3x3)
        X_3x3 = Activation('relu')(X_3x3)
        X_3x3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(X_3x3)
        X_3x3 = Conv2D(128, (3, 3), data_format='channels_last', name=name+'_inception_3b_3x3_conv2')(X_3x3)
        X_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name=name+'_inception_3b_3x3_bn2')(X_3x3)
        X_3x3 = Activation('relu')(X_3x3)

        X_5x5 = Conv2D(32, (1, 1), data_format='channels_last', name=name+'_inception_3b_5x5_conv1')(X)
        X_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name=name+'_inception_3b_5x5_bn1')(X_5x5)
        X_5x5 = Activation('relu')(X_5x5)
        X_5x5 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(X_5x5)
        X_5x5 = Conv2D(64, (5, 5), data_format='channels_last', name=name+'_inception_3b_5x5_conv2')(X_5x5)
        X_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name=name+'_inception_3b_5x5_bn2')(X_5x5)
        X_5x5 = Activation('relu')(X_5x5)

        X_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format='channels_last')(X)
        X_pool = Conv2D(64, (1, 1), data_format='channels_last', name=name+'_inception_3b_pool_conv')(X_pool)
        X_pool = BatchNormalization(axis=3, epsilon=0.00001, name=name+'_inception_3b_pool_bn')(X_pool)
        X_pool = Activation('relu')(X_pool)
        X_pool = ZeroPadding2D(padding=(4, 4), data_format='channels_last')(X_pool)

        X_1x1 = Conv2D(64, (1, 1), data_format='channels_last', name=name+'_inception_3b_1x1_conv')(X)
        X_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name=name+'_inception_3b_1x1_bn')(X_1x1)
        X_1x1 = Activation('relu')(X_1x1)

        inception = Concatenate(3)([X_3x3, X_5x5, X_pool, X_1x1])

        return inception

    def inception_block_1c(self, X, name):
        X_3x3 = utils.conv2d_bn(X,
                            name,
                            layer='inception_3c_3x3',
                            cv1_out=128,
                            cv1_filter=(1, 1),
                            cv2_out=256,
                            cv2_filter=(3, 3),
                            cv2_strides=(2, 2),
                            padding=(1, 1))

        X_5x5 = utils.conv2d_bn(X,
                            name,
                            layer='inception_3c_5x5',
                            cv1_out=32,
                            cv1_filter=(1, 1),
                            cv2_out=64,
                            cv2_filter=(5, 5),
                            cv2_strides=(2, 2),
                            padding=(2, 2))

        X_pool = MaxPooling2D(pool_size=3, strides=2, data_format='channels_last')(X)
        X_pool = ZeroPadding2D(padding=((0, 1), (0, 1)), data_format='channels_last')(X_pool)

        inception = Concatenate(3)([X_3x3, X_5x5, X_pool])

        return inception

    def inception_block_2a(self, X, name):
        X_3x3 = utils.conv2d_bn(X,
                            name,
                            layer='inception_4a_3x3',
                            cv1_out=96,
                            cv1_filter=(1, 1),
                            cv2_out=192,
                            cv2_filter=(3, 3),
                            cv2_strides=(1, 1),
                            padding=(1, 1))
        X_5x5 = utils.conv2d_bn(X,
                            name,
                            layer='inception_4a_5x5',
                            cv1_out=32,
                            cv1_filter=(1, 1),
                            cv2_out=64,
                            cv2_filter=(5, 5),
                            cv2_strides=(1, 1),
                            padding=(2, 2))

        X_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format='channels_last')(X)
        X_pool = utils.conv2d_bn(X_pool,
                            name,
                            layer='inception_4a_pool',
                            cv1_out=128,
                            cv1_filter=(1, 1),
                            padding=(2, 2))
        X_1x1 = utils.conv2d_bn(X,
                            name,
                            layer='inception_4a_1x1',
                            cv1_out=256,
                            cv1_filter=(1, 1))
        inception =Concatenate(3)([X_3x3, X_5x5, X_pool, X_1x1])

        return inception

    def inception_block_2b(self, X, name):
        #inception4e
        X_3x3 = utils.conv2d_bn(X,
                            name,
                            layer='inception_4e_3x3',
                            cv1_out=160,
                            cv1_filter=(1, 1),
                            cv2_out=256,
                            cv2_filter=(3, 3),
                            cv2_strides=(2, 2),
                            padding=(1, 1))
        X_5x5 = utils.conv2d_bn(X,
                            name,
                            layer='inception_4e_5x5',
                            cv1_out=64,
                            cv1_filter=(1, 1),
                            cv2_out=128,
                            cv2_filter=(5, 5),
                            cv2_strides=(2, 2),
                            padding=(2, 2))
        
        X_pool = MaxPooling2D(pool_size=3, strides=2, data_format='channels_last')(X)
        X_pool = ZeroPadding2D(padding=((0, 1), (0, 1)), data_format='channels_last')(X_pool)

        inception = Concatenate(3)([X_3x3, X_5x5, X_pool])

        return inception

    def inception_block_3a(self, X, name):
        X_3x3 = utils.conv2d_bn(X,
                            name,
                            layer='inception_5a_3x3',
                            cv1_out=96,
                            cv1_filter=(1, 1),
                            cv2_out=384,
                            cv2_filter=(3, 3),
                            cv2_strides=(1, 1),
                            padding=(1, 1))
        X_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format='channels_last')(X)
        X_pool = utils.conv2d_bn(X_pool,
                            name,
                            layer='inception_5a_pool',
                            cv1_out=96,
                            cv1_filter=(1, 1),
                            padding=(1, 1))
        X_1x1 = utils.conv2d_bn(X,
                            name,
                            layer='inception_5a_1x1',
                            cv1_out=256,
                            cv1_filter=(1, 1))

        inception = Concatenate(3)([X_3x3, X_pool, X_1x1])

        return inception

    def inception_block_3b(self, X, name):
        X_3x3 = utils.conv2d_bn(X,
                            name,
                            layer='inception_5b_3x3',
                            cv1_out=96,
                            cv1_filter=(1, 1),
                            cv2_out=384,
                            cv2_filter=(3, 3),
                            cv2_strides=(1, 1),
                            padding=(1, 1))
        X_pool = MaxPooling2D(pool_size=3, strides=2, data_format='channels_last')(X)
        X_pool = utils.conv2d_bn(X_pool,
                            name,
                            layer='inception_5b_pool',
                            cv1_out=96,
                            cv1_filter=(1, 1))
        X_pool = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(X_pool)

        X_1x1 = utils.conv2d_bn(X,
                            name,
                            layer='inception_5b_1x1',
                            cv1_out=256,
                            cv1_filter=(1, 1))
        inception = Concatenate(3)([X_3x3, X_pool, X_1x1])

        return inception

    def inception_model(self, X_input, name):

        X = ZeroPadding2D((3, 3))(X_input)
        
        # First Block
        X = Conv2D(64, (7, 7), strides = (2, 2), name = name+'_conv1')(X)
        X = BatchNormalization(axis=3, name = name+'_bn1')(X)
        X = Activation('relu')(X)
        
        # Zero-Padding + MAXPOOL
        X = ZeroPadding2D((1, 1))(X)
        X = MaxPooling2D((3, 3), strides = 2)(X)
        
        # Second Block
        X = Conv2D(64, (1, 1), strides = (1, 1), name = name+'_conv2')(X)
        X = BatchNormalization(axis=3, epsilon=0.00001, name = name+'_bn2')(X)
        X = Activation('relu')(X)
        
        # Zero-Padding + MAXPOOL
        X = ZeroPadding2D((1, 1))(X)

        # Second Block
        X = Conv2D(192, (3, 3), strides=(1, 1), name = name+'_conv3')(X)
        X = BatchNormalization(axis=3, epsilon=0.00001, name = name+'_bn3')(X)
        X = Activation('relu')(X)
        
        # Zero-Padding + MAXPOOL
        X = ZeroPadding2D((1, 1))(X)
        X = MaxPooling2D(pool_size=3, strides=2)(X)

        # Inception 1: a/b/c
        X = self.inception_block_1a(X, name)
        X = self.inception_block_1b(X, name)
        X = self.inception_block_1c(X, name)

        # Inception 2: a/b
        X = self.inception_block_2a(X, name)
        X = self.inception_block_2b(X, name)

        # Inception 3: a/b
        X = self.inception_block_3a(X, name)
        X = self.inception_block_3b(X, name)

        # Top Layer
        X = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(X)
        X = Flatten()(X)
        X = Dense(128, name=name+'_dense_layer')(X)

        X = Lambda(lambda x: K.l2_normalize(x, axis=1))(X)

        model = Model(inputs=X_input, outputs=X, name='FaceRecoModel')

        return model

    def get_fr_model(self):

        return self.model

    def define_training_model(self):

        anchor_input = Input(self.input_shape)
        positive_input = Input(self.input_shape)
        negative_input = Input(self.input_shape)

        model = self.model
        # tf.keras.utils.plot_model(model, to_file='model.png', show_shapes='True')

        anchor_output = tf.expand_dims(model(anchor_input), 1)
        positive_output = tf.expand_dims(model(positive_input), 1)
        negative_output = tf.expand_dims(model(negative_input), 1)

        output = Concatenate(axis=1)([anchor_output, positive_output, negative_output])

        train_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=output, name='FaceRecoTrainer')
        # tf.keras.utils.plot_model(train_model, to_file='trainer.png', show_shapes=True)

        learning_schedule = ExponentialDecay(initial_learning_rate=1e-4, decay_steps=10*8, decay_rate=0.5, staircase=True)
        optimizer = Adam(learning_rate=learning_schedule)
        train_model.compile(loss=triplet_loss, optimizer=optimizer)

        self.training_model =  train_model

    def get_training_model(self):

        return self.training_model

    def train_on_batch(self, X, y=None, sample_weight=None, class_weight=None, reset_metrics=True, return_dict=False):

        loss = self.training_model.train_on_batch(X, y, sample_weight, class_weight, reset_metrics, return_dict)

        return loss

    def img_to_encoding(self, image):

        return self.model.predict(np.array([image]))

    def recognize(self, encodings, image):
        
        closeness = None
        best_similarity = 1e+3

        image_encoded = self.model.predict(np.array([image]))
        for person, references in encodings.items():
            for reference in references:
                current_similarity = np.linalg.norm(image_encoded-reference)

                if best_similarity>current_similarity:
                    best_similarity = current_similarity
                    closeness = person

        return (closeness, best_similarity)

    def save_weights(self, path):

        self.model.save_weights(path)

    def load_weights(self, path):

        self.model.load_weights(path)
