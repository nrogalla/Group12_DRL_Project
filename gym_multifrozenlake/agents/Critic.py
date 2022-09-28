import tensorflow as tf
import numpy as np

class Critic(tf.keras.Model):
    
    def __init__(self, map_size: int, size_first_layer: int, size_second_layer: int, size_third_layer: int = 0):

        super(Critic, self).__init__()
        
        self.first_layer = tf.keras.layers.Dense(size_first_layer, activation = tf.nn.sigmoid)
        self.second_layer = tf.keras.layers.Dense(size_second_layer, activation = tf.nn.sigmoid)
        self.conv_one = tf.keras.layers.Conv2D(32, 3, activation=tf.nn.sigmoid, input_shape=(map_size, map_size, 3, 1))
        self.flatten = tf.keras.layers.GlobalMaxPooling2D()
        if (size_third_layer > 0):
            self.third_layer = tf.keras.layers.Dense(size_third_layer, activation = tf.nn.sigmoid)
        self.output_layer = tf.keras.layers.Dense(1, activation = None)

    def call(self, input):
        print("input")
        print(input)
        #input = tf.exp
        # and_dims(input, -1)
        input[1] = tf.cast(input[1], dtype = tf.int32)
        x1 = self.first_layer(input[0])
        x2 = self.conv_one(input[1])
        x2 = self.flatten(x2)
        x = tf.layers.concatenate(x1, x2)
        x = self.second_layer(x)
        
        if 'third_layer' in locals():
            x = self.third_layer(x)
            x = self.output_layer(x)
        else:
            x = self.output_layer(x)
        return x