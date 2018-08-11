'''
Loads data and trains the neural network
'''

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

#from model import Model
from model_cnn import Model

# Reproducibility
tf.set_random_seed(42)

INPUT_FILE = './data/processed.npy'
X = np.load(INPUT_FILE)
X /= 255

EPOCHS = 10000
STEP = 100


def plot(i, d_array):
    '''
    Plots the bar graphs
    '''
    plt.figure(i)

    d_array = np.stack(d_array)

    plt.plot(d_array)
    plt.legend(['Option %d'%(i+1) for i in range(8)], loc='upper left')


INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# INDICES = [9,10]

for index, sample in enumerate(X[INDICES]):
    tf.reset_default_graph()
    with tf.Session() as sess:
        M = Model(sess, logdir='./logs/' + str(INDICES[index]))
        print('Training Sample ' + str(INDICES[index]))
        D_ARRAY = []
        for epoch in range(0, EPOCHS + 1, STEP):

            if epoch < EPOCHS / 2:
                M.fit_autoencoder(sess, sample, STEP)
            else:
                M.fit(sess, sample, STEP)

            if epoch % STEP == 0:
                print(epoch)
                D_ARRAY.append(M.save_summaries(sess, sample, epoch))
        plot(INDICES[index], D_ARRAY)
        D_ARRAY = []
        M.close()
plt.show()
