from util import get_data
from variables import*
import os
from model import IrisClassifier
import numpy as np
from sklearn.utils import shuffle

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import logging
logging.getLogger('tensorflow').disabled = True

current_dir = os.getcwd()
saved_weights = os.path.join(current_dir,saved_weights)
classifier = IrisClassifier()

if __name__ == "__main__":
    Xtrain, Xtest, Ytrain, Ytest = get_data()
    Xtest, Ytest = shuffle(Xtest, Ytest)
    if os.path.exists(saved_weights):
        print("Loading existing model !!!")
        classifier.load_model()
    else:
        print("Training the model  and saving!!!")
        classifier.mnist_model()
        classifier.train()
        classifier.save_model()

    # Define a generator function that provides our test data's x values
    # as a representative dataset, and tell the converter to use it
    def representative_dataset_generator():
        for value in Xtest:
            yield [np.array(value, dtype=np.float32, ndmin=2)]

    if not os.path.exists(tflite_path):
        # Convert the model to the TensorFlow Lite format without quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(classifier.model)
        tflite_model = converter.convert()

        # Save the model to disk
        open(tflite_path, "wb").write(tflite_model)

    if not os.path.exists(quantized_tflite_path):
        # Convert the model to the TensorFlow Lite format with quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(classifier.model)

        # Indicate that we want to perform the default optimizations,
        # which include quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_generator
        # Convert the model
        tflite_model = converter.convert()
        # Save the model to disk
        open(quantized_tflite_path , "wb").write(tflite_model)