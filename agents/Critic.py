import tensorflow as tf
import numpy as np

class Critic(tf.keras.Model):
    
    def __init__(self, map_size: int, size_first_layer: int, size_second_layer: int, size_third_layer: int = 0):

        super(Critic, self).__init__()
        
        self.first_layer = tf.keras.layers.Dense(size_first_layer, activation = tf.nn.sigmoid)
        self.second_layer = tf.keras.layers.Dense(size_second_layer, activation = tf.nn.sigmoid)
        self.conv_one = tf.keras.layers.Conv2D(32, 3, activation=tf.nn.relu, padding = 'same', input_shape=(map_size, map_size, 3))
        self.flatten = tf.keras.layers.Flatten()
        if (size_third_layer > 0):
            self.third_layer = tf.keras.layers.Dense(size_third_layer, activation = tf.nn.sigmoid)
        self.output_layer = tf.keras.layers.Dense(1, activation = None)

    def call(self, input):
        
        input_conv = tf.cast(input[1], dtype = tf.float32)
        x1 = self.first_layer(input[0])
        x2 = self.conv_one(input_conv)
        x2 = self.flatten(x2)
        x = tf.keras.layers.concatenate((x1, x2))
        x = self.second_layer(x) 
        if 'third_layer' in locals():
            x = self.third_layer(x)
        x = self.output_layer(x)
        return x