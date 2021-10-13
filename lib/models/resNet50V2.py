from tensorflow.keras.layers import *
import tensorflow as tf

from . import top

def get_model(input_shape, num_outputs):
    model = tf.keras.applications.ResNet50V2(include_top=False, input_shape=input_shape, pooling='avg')

    return top.add_top(model, num_outputs)