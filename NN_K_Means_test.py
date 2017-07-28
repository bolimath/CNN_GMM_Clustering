import tensorflow as tf
from sklearn import datasets
from sklearn.cluster import *


import matplotlib.pyplot as plt

import numpy as np

iris = datasets.load_iris()
X = iris.data


class NN:
    def __init__(self):

        self.sess = tf.InteractiveSession()

        # Make the neural network
        self.input = tf.placeholder(tf.float32, [None, 4])

        W1 = tf.Variable(tf.random_normal([4,6], stddev=0.1))
        b1 = tf.Variable(tf.zeros([6]))
        h = tf.nn.sigmoid(tf.matmul(self.input, W1) + b1)
        W2 = tf.Variable(tf.random_normal([6,2], stddev=0.1))
        b2 = tf.Variable(tf.zeros([2]))

        self.out = tf.matmul(h, W2) + b2
        self.tgt = tf.placeholder(tf.float32, [None, 2])
        self.loss = tf.reduce_mean(tf.square(self.out - self.tgt))
        self.train = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())


    def train(self, _x, _y):
        _, loss = self.sess.run([self.train, self.loss], feed_dict={self.input: _x, self.tgt: _y})
        return loss

    def fwd(self, _x):
        return self.sess.run(self.out, feed_dict={self.input: _x})


def display(clusters, cluster_centers, colors=['r','b','g']):
    """
    """

    plt.hold(True)

    for i in range(3):
        cluster = clusters[i]
        center = cluster_centers[i]
        color = colors[i]

        plt.plot(cluster[:,0], cluster[:,1], 'o'+color)
        plt.plot(center[0], center[1], '+'+color)

    plt.show()


def cluster(y, num_clusters=3):
    """
    Perform clustering on y
    """

    km = KMeans(num_clusters)
    km.fit(y)
    centers = km.cluster_centers_
    labels = km.labels_

    return centers, labels

def step(_x, nn):
    # 1.  Do a forward pass
    y = nn.fwd(_x)

    # 2. Perform clustering
    centers, labels = cluster(y)

    # 3. Repel the centers

