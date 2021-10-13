from tensorflow.keras.layers import *
import tensorflow as tf

def add_top(model, num_outputs):

    completed_model = tf.keras.Sequential([
        model,
        Flatten(),
        Dense(num_outputs, 'softmax')
    ])

    return completed_model