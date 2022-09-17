import tensorflow as tf
import numpy as np

class Actor(tf.keras.Model):
    
    def __init__(self, number_actions: int, number_observations: int, size_first_layer: int, size_second_layer: int, size_third_layer: int = 0):

        super(Actor, self).__init__()

        self.input_layer = tf.keras.layers.Input(shape(number_observations, ))
        self.first_layer = tf.keras.layers.Dense(size_first_layer, activation = tf.nn.relu)
        self.second_layer = tf.keras.layers.Dense(size_second_layer, activation = tf.nn.relu)
        if (size_third_layer > 0):
            self.third_layer = tf.keras.layers.Dense(size_third_layer, activation = tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(number_actions, activation = tf.nn.softmax)

    def call(self, input):

        x = self.input_layer(input)
        x = self.first_layer(x)
        x = self.second_layer(x)
        if 'third_layer' in locals():
            x = self.third_layer(x)
            x = self.output_layer(x)
        else:
            x = self.output_layer(x)
        return x





