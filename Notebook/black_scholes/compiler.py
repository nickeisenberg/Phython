import numpy as np
from keras.utils import timeseries_dataset_from_array as ts
from keras.backend import get_value
import tensorflow as tf

data = np.linspace(0, 10, 11)
dataset = ts(data=data, targets=None, sequence_length=3, sequence_stride=1,
             batch_size=None)


for d in dataset:
    print(np.array(d).std())

