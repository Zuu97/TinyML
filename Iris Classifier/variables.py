csv_path = 'data/iris.csv'
label_encode = {'setosa':0,
                'versicolor':1,
                'virginica':2
                }
file_name = 'data/iris2.csv'
cutoff = 0.8
seed = 42
dense1 = 64
sense2 = 32
output = 3
tensor_shape = 4
num_epochs = 120
saved_weights = 'data/irish_weights.h5'
tflite_path = "data/irish_model.tflite"
quantized_tflite_path = "data/quantized_irish_model.tflite"