import tensorflow as tf
import numpy as np

class Actor(tf.keras.Model):
    
    def __init__(self, number_actions: int, number_observations: int, size_first_layer: int, size_second_layer: int, size_third_layer: int = 0):

        super(Actor, self).__init__()       
        self.first_layer = tf.keras.layers.Dense(size_first_layer, activation = tf.nn.relu, kernel_initializer = 'zeros')
        self.second_layer = tf.keras.layers.Dense(size_second_layer, activation = tf.nn.relu, kernel_initializer = 'zeros')
        if (size_third_layer > 0):
            self.third_layer = tf.keras.layers.Dense(size_third_layer, activation = tf.nn.relu, kernel_initializer = 'zeros')
        self.output_layer = tf.keras.layers.Dense(number_actions, activation = tf.nn.softmax, kernel_initializer = 'zeros')

    def call(self, input):
        
        input = tf.expand_dims(input, -1)
        x = self.first_layer(input)
        x = self.second_layer(x)
        if 'third_layer' in locals():
            x = self.third_layer(x)
            x = self.output_layer(x)
        else:
            x = self.output_layer(x)
        return x





