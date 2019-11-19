# Following https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# and https://keras.io/utils/

import numpy as np
import keras
import tensorflow as tf

class DataGenerator(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x = x_set
        # This should be the ragged tensor
        self.y = y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y
