from util import get_data
from variables import*
import os
from model import IrisClassifier
import numpy as np
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import logging
logging.getLogger('tensorflow').disabled = True

current_dir = os.getcwd()
saved_weights = os.path.join(current_dir,saved_weights)
classifier = IrisClassifier()
classifier.run()

def representative_dataset_generator():
    for value in classifier.Xtest:
        yield [np.array(value, dtype=np.float32, ndmin=2)]

def tflite_model(converter):
    if not os.path.exists(tflite_path):
        tflite_model = converter.convert() # Convert the model
        open(tflite_path, "wb").write(tflite_model)

    if not os.path.exists(quantized_tflite_path):
        converter.optimizations = [tf.lite.Optimize.DEFAULT] # Set parameters to default
        converter.representative_dataset = representative_dataset_generator
        tflite_model = converter.convert() # Convert the model
        open(quantized_tflite_path , "wb").write(tflite_model)

    basic_model_size = os.path.getsize(tflite_path)
    print("Non Quantized model size : {} bytes".format(basic_model_size))
    quantized_model_size = os.path.getsize(quantized_tflite_path)
    print("Quantized model size : {} bytes\n".format(quantized_model_size))

def predict_from_non_quantized_model():
    model = tf.lite.Interpreter(tflite_path)
    model.allocate_tensors() # allocate memory
    model_input_index = model.get_input_details()[0]["index"]
    model_output_index = model.get_output_details()[0]["index"]

    model_predictions = []
    n_correct = 0
    for x_value, y_value in zip(classifier.Xtest,classifier.Ytest):
        x_value_tensor = tf.convert_to_tensor([x_value], dtype=np.float32) # Create a 2D tensor wrapping the current x value
        model.set_tensor(model_input_index, x_value_tensor) # Write the value to the input tensor
        model.invoke() # Run inference
        p = model.get_tensor(model_output_index)[0].argmax()
        model_predictions.append(p)

        n_correct = n_correct + 1 if p == y_value else n_correct
    accuracy = n_correct/len(classifier.Ytest)
    print("accuracy : ",accuracy)
    print("Predictions : ",model_predictions)

def predict_from_quantized_model():
    quantized_model = tf.lite.Interpreter(quantized_tflite_path)
    quantized_model.allocate_tensors() # allocate memory
    quantized_model_input_index = quantized_model.get_input_details()[0]["index"]
    quantized_model_output_index = quantized_model.get_output_details()[0]["index"]

    model_predictions = []
    n_correct = 0
    for x_value, y_value in zip(classifier.Xtest,classifier.Ytest):
        x_value_tensor = tf.convert_to_tensor([x_value], dtype=np.float32) # Create a 2D tensor wrapping the current x value
        quantized_model.set_tensor(quantized_model_input_index, x_value_tensor) # Write the value to the input tensor
        quantized_model.invoke() # Run inference
        p = quantized_model.get_tensor(quantized_model_output_index)[0].argmax()
        model_predictions.append(p)

        n_correct = n_correct + 1 if p == y_value else n_correct
    accuracy = n_correct/len(classifier.Ytest)
    print("Quantized Accuracy : ",accuracy)
    print("Quantized Predictions : ",model_predictions)

def deployment():
    # Convert the model to the TensorFlow Lite format with or without quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(classifier.model)
    tflite_model(converter)
    predict_from_non_quantized_model()
    predict_from_quantized_model()