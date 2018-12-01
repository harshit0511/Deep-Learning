import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data

class DeepConvNet:
    def __init__(self, input_data, input_shape = (28, 28, 1), loss = 'categorical_crossentropy',
                 lr = 1e-3, metrics = ['accuracy'], batch_size=128, epochs=20):
        """
        Initializing variables and data
        """
        # Model parameters
        self.input_shape = input_shape
        self.learning_rate = lr
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs
        # Initializing Data
        self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        self.x_train, self.x_test, self.y_train, self.y_test = self.get_data()
        # Initializing the model
        self.model = self.model()

    def get_data(self):
        """
        returns MNIST data by splitting into train/test
        """
        x_train = self.mnist.train.images
        x_test = self.mnist.test.images
        y_train = self.mnist.train.labels
        y_test = self.mnist.test.labels

        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        return x_train, x_test, y_train, y_test

    def model(self):
        """
        Specifying the architecture of the network using Keras high-level API
        """
        inputs = keras.Input(shape=self.input_shape)
        # First convolutional layer
        x = keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu') (inputs)
        x = keras.layers.MaxPool2D(pool_size=(2, 2)) (x)
        x = keras.layers.Dropout(0.5) (x)
        # Second convolutional layer
        x = keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu') (x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2)) (x)
        x = keras.layers.Dropout(0.5) (x)
        # Fully-connected layer
        x = keras.layers.Flatten() (x)
        x = keras.layers.Dense(32, 'relu') (x)
        x = keras.layers.Dropout(0.5) (x)
        x = keras.layers.Dense(16) (x)
        x = keras.layers.BatchNormalization() (x)
        x = keras.layers.Activation('relu') (x)
        preds = keras.layers.Dense(10, 'softmax') (x)
        model = keras.Model(inputs=inputs, outputs=preds)
        print (model.summary())
        return model

    def train(self):
        """
        Compile and train the model
        """
        self.model.compile(optimizer=tf.train.AdamOptimizer(self.learning_rate),
                          loss = self.loss,
                          metrics = self.metrics)

        self.model.fit(x=self.x_train, y=self.y_train, batch_size=self.batch_size,
                      epochs = self.epochs,
                      validation_data=(self.x_test, self.y_test))

if __name__ == '__main__':
    CNN = DeepConvNet(input_data)
    CNN.train()
