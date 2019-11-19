import tensorflow as tf
import keras
from dummygenerator import DataGenerator


# From https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
# Define custom loss

# Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
def loss(y_true,y_pred):
    return keras.backend.square(y_pred[0] - y_true[0]) 

generator = DataGenerator(tf.constant([[1,2,3], [4, 5, 6]]), tf.ragged.constant([[1, 2, 3], [4, 5]]), 1)

model = keras.models.Sequential()
model.add(keras.layers.Dense(12, input_dim=2, activation='relu'))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss=loss)

model.fit_generator(generator=generator, validation_data=generator)
