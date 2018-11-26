import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

class MultiLayerPerceptron:
    def __init__(self, input_data, lr=1e-3, num_steps=6000, batch_size=128, n_hidden_1=256,
                n_hidden_2=256, num_input=784, num_classes=10):
        """
        Initializing variables
        """
        # Parameters
        self.learning_rate = lr
        self.num_steps = num_steps
        self.batch_size = batch_size
        #self.display_step = 500
        # Network Parameters
        self.n_hidden_1 = n_hidden_1 # 1st layer number of neurons
        self.n_hidden_2 = n_hidden_2 # 2nd layer number of neurons
        self.num_input = num_input # MNIST data input (img shape: 28*28)
        self.num_classes = num_classes # MNIST total classes (0-9 digits)
        # tf Graph input
        self.X = tf.placeholder("float", [None, self.num_input])
        self.Y = tf.placeholder("float", [None, self.num_classes])
        self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        """
        Define the loss function and optimizer
        """
        logits = self.neural_net()

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                    logits=logits, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def get_batch(self, start, end):
        x = self.mnist.train.images[start:end]
        y = self.mnist.train.labels[start:end]
        return x, y

    def parameters(self):
        """
        Initialize the weights and biases with given shapes
        """
        weights = {
            'h1': tf.Variable(tf.random_normal([self.num_input, self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.num_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.num_classes]))
        }
        return weights, biases

    def neural_net(self):
        """
        Constructing the architecture of the network
        """
        weights, biases = self.parameters()
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(self.X, weights['h1']), biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
        return out_layer

    def train(self):
        """
        Train the network with the MNIST data
        """
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            start = 0
            end = self.batch_size
            sess.run(init)
            for step in np.arange(1, self.num_steps+1):
                #print (start, end)
                batch_x, batch_y = self.get_batch(start, end)
                #batch_x, batch_y = mnist.train.next_batch(batch_size)
                sess.run(self.train_op, feed_dict={self.X: batch_x, self.Y: batch_y})
                start += self.batch_size
                end += self.batch_size

                if step % 500 == 0 or step == 1:
                    loss, acc = sess.run([self.loss_op, self.accuracy], feed_dict={self.X:batch_x,
                                                                        self.Y:batch_y})
                    print ("Step: ", step,
                          "\nLoss: ", loss,
                          "\nAccuracy: ", acc)

                if end >= len(self.mnist.train.images):
                    start = 0
                    end = self.batch_size

            print ("Optimization complete")

            print ("Testing accuracy: ",
                  sess.run(self.accuracy, feed_dict={self.X:self.mnist.test.images,
                                               self.Y:self.mnist.test.labels}))

if __name__ == '__main__':
    MLP = MultiLayerPerceptron(input_data)
    MLP.train()
