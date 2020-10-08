from util import get_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import logging
logging.getLogger('tensorflow').disabled = True
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from variables import *
import numpy as np
from tensorflow.keras.models import load_model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\nNum GPUs Available: {}".format(len(physical_devices)))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

class IrisClassifier(object):
    def __init__(self):
        Xtrain, Xtest, Ytrain, Ytest = get_data()
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest


    def iris_model(self):
        self.model = Sequential([
                    Dense(dense1, input_shape=(tensor_shape,), activation='relu'),
                    Dense(dense1, activation='relu'),
                    Dropout(0.5),
                    Dense(output, activation='softmax'),
                ])

    def train(self):
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
                      )
        self.model.summary()
        self.model.fit(
            self.Xtrain,
            self.Ytrain,
            epochs=num_epochs,
            validation_data=(self.Xtest,self.Ytest)
            )

    def save_model(self):
        self.model.save(saved_weights)

    def load_model(self):
        loaded_model = load_model(saved_weights)
        loaded_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
                      )
        self.model = loaded_model

    def run(self):
        if os.path.exists(saved_weights):
            print("Loading existing model !!!")
            self.load_model()
        else:
            print("Training the model  and saving!!!")
            self.iris_model()
            self.train()
            self.save_model()