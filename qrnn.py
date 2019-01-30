"""
more detail can be found in the following url:
https://github.com/icoxfog417/tensorflow_qrnn
"""

import tensorflow as tf


class QRNN(object):

    def __init__(self, in_size, size, name, conv_size=2):
        self.kernel = None
        self.batch_size = -1
        self.conv_size = conv_size
        self.c = None
        self.h = None
        self.seq_h = list()
        self._x = None
        self.name = name
        if conv_size == 1:
            self.kernel = QRNNLinear(in_size, size, name=self.name)
        elif conv_size == 2:
            self.kernel = QRNNWithPrevious(in_size, size, name=self.name)
        else:
            self.kernel = QRNNConvolution(in_size, size, conv_size, name=self.name)

    def _step(self, f, z, o):
        with tf.variable_scope("fo-Pool/%s" % self.name):
            # f,z,o is batch_size x size
            f = tf.sigmoid(f)
            z = tf.tanh(z)
            o = tf.sigmoid(o)
            self.c = tf.multiply(f, self.c) + tf.multiply(1 - f, z)
            self.h = tf.multiply(o, self.c)  # h is size vector

        return self.h

    def forward(self, x, return_sequence=False):
        length = lambda mx: int(mx.get_shape()[0])

        with tf.variable_scope("QRNN/Forward/%s" % self.name):
            if self.c is None:
                # init context cell
                self.c = tf.zeros([length(x), self.kernel.size], dtype=tf.float32)

            if self.conv_size <= 2:
                # x is batch_size x sentence_length x word_length
                # -> now, transpose it to sentence_length x batch_size x word_length
                _x = tf.transpose(x, [1, 0, 2])

                for i in range(length(_x)):
                    t = _x[i]  # t is batch_size x word_length matrix
                    f, z, o = self.kernel.forward(t)
                    self.seq_h.append(self._step(f, z, o))
            else:
                c_f, c_z, c_o = self.kernel.conv(x)
                for i in range(length(c_f)):
                    f, z, o = c_f[i], c_z[i], c_o[i]
                    self.seq_h.append(self._step(f, z, o))
        if return_sequence:
            return tf.transpose(self.seq_h, [1, 0, 2])
        else:
            return self.h


class QRNNLinear(object):

    def __init__(self, in_size, size, name):
        self.in_size = in_size
        self.size = size
        self._weight_size = self.size * 3  # z, f, o
        self.name = name
        with tf.variable_scope("QRNN/Variable/Linear/%s" % self.name):
            initializer = tf.random_normal_initializer()
            self.W = tf.get_variable("W", [self.in_size, self._weight_size], initializer=initializer)
            self.b = tf.get_variable("b", [self._weight_size], initializer=initializer)

    def forward(self, t):
        # x is batch_size x word_length matrix
        _weighted = tf.matmul(t, self.W)
        _weighted = tf.add(_weighted, self.b)

        # now, _weighted is batch_size x weight_size
        f, z, o = tf.split(_weighted, num_or_size_splits=3, axis=1)  # split to f, z, o. each matrix is batch_size x size
        return f, z, o


class QRNNWithPrevious(object):

    def __init__(self, in_size, size, name):
        self.in_size = in_size
        self.size = size
        self._weight_size = self.size * 3  # z, f, o
        self._previous = None
        self.name = name
        with tf.variable_scope("QRNN/Variable/WithPrevious/%s" % self.name):
            initializer = tf.random_normal_initializer()
            self.W = tf.get_variable("W", [self.in_size, self._weight_size], initializer=initializer)
            self.V = tf.get_variable("V", [self.in_size, self._weight_size], initializer=initializer)
            self.b = tf.get_variable("b", [self._weight_size], initializer=initializer)

    def forward(self, t):
        if self._previous is None:
            self._previous = tf.get_variable("previous/%s" % self.name, [t.get_shape()[0], self.in_size], initializer=tf.random_normal_initializer())

        _current = tf.matmul(t, self.W)
        _previous = tf.matmul(self._previous, self.V)
        _previous = tf.add(_previous, self.b)
        _weighted = tf.add(_current, _previous)

        f, z, o = tf.split(_weighted, num_or_size_splits=3, axis=1)  # split to f, z, o. each matrix is batch_size x size
        self._previous = t
        return f, z, o


class QRNNConvolution(object):

    def __init__(self, in_size, size, conv_size, name):
        self.in_size = in_size
        self.size = size
        self.conv_size = conv_size
        self._weight_size = self.size * 3  # z, f, o
        self.name = name

        with tf.variable_scope("QRNN/Variable/Convolution/%s" % self.name):
            initializer = tf.random_normal_initializer()
            self.conv_filter = tf.get_variable("conv_filter/%s" % self.name, [conv_size, in_size, self._weight_size], initializer=initializer)

    def conv(self, x):
        # !! x is batch_size x sentence_length x word_length(=channel) !!
        _weighted = tf.nn.conv1d(x, self.conv_filter, stride=1, padding="SAME")

        # _weighted is batch_size x conved_size x output_channel
        _w = tf.transpose(_weighted, [1, 0, 2])  # conved_size x  batch_size x output_channel
        _ws = tf.split(_w, num_or_size_splits=3, axis=2) # make 3(f, z, o) conved_size x  batch_size x size
        return _ws


if __name__ == '__main__':
    import numpy as np

    def create_test_data(batch_size, sentence_length, word_size):
        batch = []
        for b in range(batch_size):
            sentence = np.random.rand(sentence_length, word_size)
            batch.append(sentence)
        return np.array(batch)

    batch_size = 100
    sentence_length = 5
    word_size = 10
    size = 5
    data = create_test_data(batch_size, sentence_length, word_size)

    with tf.Graph().as_default() as q_conv:
        # input layer
        X = tf.placeholder(tf.float32, [batch_size, sentence_length, word_size])

        # layer 1
        qrnn = QRNN(in_size=word_size, size=size, conv_size=3, name='1')
        hidden_1 = qrnn.forward(X)

        # reshape for layer 2
        hidden_1 = tf.reshape(hidden_1, (hidden_1.shape[0], hidden_1.shape[1], 1))

        # layer 2
        qrnn2 = QRNN(in_size=1, size=size, conv_size=3, name='2')
        output = qrnn2.forward(hidden_1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            hidden = sess.run(output, feed_dict={X: data})
            # self.assertEqual((batch_size, size), hidden.shape)
            pass
