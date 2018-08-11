'''
CNN Model for the Raven Matrix Autoencoder
'''

import tensorflow as tf
import numpy as np


class Model:
    '''
    Raven Matrix CNN Autoencoder
    '''

    def __init__(self, sess, logdir='./logs', img_shape=None, num_imgs=16):
        '''
        Constructor for the Raven Matrix Autoencoder
        '''

        if img_shape is None:
            img_shape = [64, 64]

        # Shape calculations
        self.shapes = {}
        self.shapes['img_shape'] = [1, img_shape[0], img_shape[1], 1]
        channels = [1, 8, 2]
        filter_size = [3, 3, 3]
        self.strides = [[1, 1, 1, 1], [1, 2, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

        # Placeholder
        placeholder_shape = [num_imgs, img_shape[0], img_shape[1], 1]
        self.img_placeholder = tf.placeholder(
            tf.float32, placeholder_shape, name="img_placeholder")

        # Autoencoder
        self.weights = {}
        with tf.name_scope("autoencoder_weights"):
            ae_w1_shape = [
                filter_size[0], filter_size[0], channels[0], channels[1]
            ]
            ae_w2_shape = [
                filter_size[1], filter_size[1], channels[1], channels[2]
            ]

            self.weights['AE_W1'] = tf.Variable(
                tf.random_normal(ae_w1_shape), name='W1')
            self.weights['AE_W2'] = tf.Variable(
                tf.random_normal(ae_w2_shape), name='W2')

        with tf.name_scope("autoencoder"):
            self.autoencode = self._decoder(
                self._encoder(self.img_placeholder))

        with tf.name_scope("autoencode_image_summaries"):
            for i in range(16):
                tf.summary.image('img_' + str(i),
                                 tf.reshape(self.autoencode[i],
                                            self.shapes['img_shape']))

        # Regressor weights
        with tf.name_scope("regressor_weights"):
            intermediate = 10

            r_w1_shape = [
                filter_size[2], filter_size[2], channels[2] * 2, intermediate
            ]
            r_w2_shape = [
                filter_size[2], filter_size[2], intermediate, channels[2]
            ]

            self.weights['R_W1'] = tf.Variable(
                tf.random_normal(r_w1_shape), name='W1')
            self.weights['R_W2'] = tf.Variable(
                tf.random_normal(r_w2_shape), name='W2')

        with tf.name_scope("regressor"):
            self.regress = self._get_regressor(self.img_placeholder)

        # Loss and optimization
        autoenc_loss, regress_loss = self._get_loss()
        self.opt = {}
        self.opt['autoenc'] = tf.train.AdamOptimizer().minimize(autoenc_loss)
        self.opt['regress'] = tf.train.AdamOptimizer().minimize(regress_loss)

        sess.run(tf.global_variables_initializer())

        self.summary = {}
        self.summary['op'] = tf.summary.merge_all()
        self.summary['writer'] = tf.summary.FileWriter(logdir, sess.graph)
        self.summary['output'] = None

    def _encoder(self, x_input):
        '''
        Converts tensor from image space to latent space
        '''
        with tf.name_scope('encoder'):
            layer_1 = tf.nn.conv2d(
                x_input,
                self.weights['AE_W1'],
                self.strides[0],
                padding='SAME')
            layer_1 = tf.nn.sigmoid(layer_1)
            self.shapes['layer_1'] = layer_1.get_shape().as_list()

            layer_2 = tf.nn.conv2d(
                layer_1,
                self.weights['AE_W2'],
                self.strides[1],
                padding='SAME')
            layer_2 = tf.nn.sigmoid(layer_2)
            self.shapes['layer_2'] = layer_2.get_shape().as_list()
        return layer_2

    def _decoder(self, z_input):
        '''
        Converts tensor from latent space to image space
        '''
        with tf.name_scope('decoder'):
            batch_size = tf.shape(z_input)[0]

            l1_size = [
                batch_size, self.shapes['layer_1'][1],
                self.shapes['layer_1'][2], self.shapes['layer_1'][3]
            ]
            l2_size = [
                batch_size, self.shapes['img_shape'][1],
                self.shapes['img_shape'][2], self.shapes['img_shape'][3]
            ]

            layer_1 = tf.nn.conv2d_transpose(
                z_input,
                self.weights['AE_W2'],
                l1_size,
                self.strides[1],
                padding='SAME')
            layer_1 = tf.nn.sigmoid(layer_1)
            layer_2 = tf.nn.conv2d_transpose(
                layer_1,
                self.weights['AE_W1'],
                l2_size,
                self.strides[0],
                padding='SAME')
            layer_2 = tf.nn.sigmoid(layer_2)
        return layer_2

    def _get_regressor(self, x_input):
        '''
        Gets regression outputs
        '''
        # Get the first two examples
        with tf.name_scope("gather_tensors"):
            x_1 = tf.gather(x_input, [0, 3, 6])
            x_2 = tf.gather(x_input, [1, 4, 7])

        # Get the encoded results
        with tf.name_scope("encode_gathered"):
            z_1 = self._encoder(x_1)
            z_2 = self._encoder(x_2)

        # Concat into a single vector
        z_c = tf.concat([z_1, z_2], axis=3, name='concatenate_Z')

        # Predict the latent space of the result
        with tf.name_scope("regression"):
            layer_1 = tf.nn.conv2d(
                z_c, self.weights['R_W1'], self.strides[2], padding='SAME')
            layer_1 = tf.nn.sigmoid(layer_1)

            layer_2 = tf.nn.conv2d(
                layer_1, self.weights['R_W2'], self.strides[3], padding='SAME')
            layer_2 = tf.nn.sigmoid(layer_2)

        # Decode the latent spaces
        regressed_imgs = self._decoder(layer_2)

        # Image summaries
        with tf.name_scope("image_summaries"):
            tf.summary.image('a3',
                             tf.reshape(regressed_imgs[0],
                                        self.shapes['img_shape']))
            tf.summary.image('b3',
                             tf.reshape(regressed_imgs[1],
                                        self.shapes['img_shape']))
            tf.summary.image('c3',
                             tf.reshape(regressed_imgs[2],
                                        self.shapes['img_shape']))

        return regressed_imgs

    def _get_loss(self):
        '''
        Gets the graph for autoencoder loss and regression loss
        '''
        # Autoencode loss
        with tf.name_scope("autoencode_loss"):
            mse = tf.reduce_mean(
                tf.squared_difference(self.autoencode, self.img_placeholder))
            #psnr = 10 * tf.log(mse) / np.log(10)
            autoencode_loss = mse
            tf.summary.scalar('autoencode_loss', autoencode_loss)

        # Regression loss
        with tf.name_scope("regress_loss"):
            regress_result = tf.gather(self.regress, [0, 1])
            expected_output = tf.gather(self.img_placeholder, [2, 5])
            mse = tf.reduce_mean(
                tf.squared_difference(regress_result, expected_output))
            #psnr = 10 * tf.log(mse) / np.log(10)
            regress_loss = mse
            tf.summary.scalar('regress_loss', regress_loss)

        return autoencode_loss, regress_loss

    def fit(self, sess, input_data, steps):
        '''
        Fits all supplied data by one learning step
        '''
        input_data = input_data.reshape([16, 64, 64, 1])
        fetches = [self.opt['autoenc'], self.opt['regress']]
        feed_dict = {self.img_placeholder: input_data}
        for _ in range(steps):
            sess.run(fetches, feed_dict)

    def fit_autoencoder(self, sess, input_data, steps):
        '''
        Only fit the autoencoder
        '''
        input_data = input_data.reshape([16, 64, 64, 1])
        fetches = [self.opt['autoenc']]
        feed_dict = {self.img_placeholder: input_data}
        for _ in range(steps):
            sess.run(fetches, feed_dict)

    def save_summaries(self, sess, input_data, epoch):
        '''
        Writes the tf.summary to disk
        '''
        input_data = input_data.reshape([16, 64, 64, 1])
        feed_dict = {self.img_placeholder: input_data}
        fetches = [self.regress, self.summary['op']]

        regression, summary_output = sess.run(fetches, feed_dict)

        self.summary['writer'].add_summary(summary_output, epoch)
        
        return self._calculate_distances(input_data, regression)

    def _calculate_distances(self, actual, regression):
        distances = np.zeros(8)
        for i in range(8):
            mse = np.average(np.square(actual[i + 8] - regression[2]))
            psnr = -10 * np.log10(mse)
            distances[i] = psnr
        return distances

    def close(self):
        '''
        Cleanup
        '''
        print('Done')
        self.summary['writer'].close()
