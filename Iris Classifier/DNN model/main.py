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
from tflite import *

if __name__ == "__main__":
    deployment()
