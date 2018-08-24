import numpy as np
import tensorflow as tf
import CCR.utils as utils
from tensorflow.python.training import moving_averages
import CCR.ocr_mtrics as ocr_mtrics
from tensorflow.contrib import slim
import CCR.spatial_transformer as st

FLAGS = utils.FLAGS
num_classes = utils.num_classes


class LSTMOCR(object):
    def __init__(self, mode):
        self.mode = mode
        # image
        self.inputs = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel], name="ccr_model_input")
        # SparseTensor required by ctc_loss op
        self.labels = tf.sparse_placeholder(tf.int32)
        # 1d array of size [batch_size]
        # l2
        self._extra_train_ops = []

    def build_graph(self):
        self._build_model()
        self._build_train_op()
        self.__build_eval_node()
        self.merged_summay = tf.summary.merge_all()
        return

    def __build_eval_node(self):
        predictions = self.dense_decoded
        labels = tf.sparse_tensor_to_dense(self.labels, default_value=-1)
        print(predictions)
        if self.mode == 'train':
            character_acc = ocr_mtrics.character_accuracy(predictions, labels)
            word_acc = ocr_mtrics.word_accuracy(predictions, labels)
            character_acc = tf.Print(character_acc, [character_acc], "character_accuracy")
            tf.summary.scalar('train_character_accuracy', character_acc)
            word_acc = tf.Print(word_acc, [word_acc], "word_acc")
            tf.summary.scalar('train_word_acc', word_acc)
            tf.summary.scalar('loss', self.cost)
            return

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'eval/characteracc': ocr_mtrics.streaming_character_accuracy(predictions, labels),
            'eval/wordacc': ocr_mtrics.streaming_word_accuracy(predictions, labels)
        })
        for metric, value in names_to_values.items():
            value = tf.Print(value, [value], metric)
            tf.summary.scalar(metric, value)
        self.names_to_updates = list(names_to_updates.values())
        return

    def _build_model(self):
        filters = [64, 128, 128, FLAGS.max_stepsize]
        strides = [1, 2]
        theta = np.array([[1., 0, 0], [0, 1., 0]])

        with tf.variable_scope('stn-1'):
            x = self._spatial_transform(self.inputs)

        with tf.variable_scope('cnn'):
            with tf.variable_scope('unit-1'):
                x = self._conv2d(x, 'cnn-1', (3, 3), 3, filters[0], strides[0])
                x = self._batch_norm('bn1', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, 2, strides[1])

            with tf.variable_scope('unit-2'):
                x = self._conv2d(x, 'cnn-2', (3, 3), filters[0], filters[1], strides[0])
                x = self._batch_norm('bn2', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, 2, strides[1])

#            with tf.variable_scope('stn-2'):
#                x = self._spatial_transform(x)

            with tf.variable_scope('unit-3'):
                x = self._conv2d(x, 'cnn-3', (3, 5), filters[1], filters[2], strides[0])
                x = self._batch_norm('bn3', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, 2, strides[1])

            with tf.variable_scope('unit-6'):
                x = self._conv2d(x, 'cnn-4', (3, 5), filters[2], filters[3], strides[0])
                x = self._batch_norm('bn4', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, 2, strides[1])

#        with tf.variable_scope('stn-2'):
#            x = self._spatial_transform(x)

        with tf.variable_scope('lstm'):
            # [batch_size, max_stepsize, num_features]
            batch_size, height, width, channels = x.get_shape().as_list()
            x = tf.transpose(x, [0, 2, 1, 3]) 
            x = tf.reshape(x, [-1, width, height * channels])
            self.seq_len = tf.fill([tf.shape(x)[0]], width)

            # tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.GRUCell
            cell = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            if self.mode == 'train':
                cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.8)

            cell1 = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            if self.mode == 'train':
                cell1 = tf.contrib.rnn.DropoutWrapper(cell=cell1, output_keep_prob=0.8)

            # Stacking rnn cells
            stack = tf.contrib.rnn.MultiRNNCell([cell, cell1], state_is_tuple=True)

            # The second output is the last state and we will not use that
            outputs, _ = tf.nn.dynamic_rnn(stack, x, self.seq_len, dtype=tf.float32)

            # Reshaping to apply the same weights over the timesteps
            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden])

            W = tf.get_variable(name='W',
                                shape=[FLAGS.num_hidden, num_classes],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='b',
                                shape=[num_classes],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            self.logits = tf.matmul(outputs, W) + b
            # Reshaping back to the original shape
            shape = tf.shape(x)
            self.logits = tf.reshape(self.logits, [shape[0], -1, num_classes])
            # Time major
            self.logits = tf.transpose(self.logits, (1, 0, 2))


    def _build_train_op(self):
        self.global_step = tf.Variable(0, trainable=False)

        self.loss = tf.nn.ctc_loss(labels=self.labels,
                                   inputs=self.logits,
                                   sequence_length=self.seq_len)
        self.cost = tf.reduce_mean(self.loss)

        self.lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                   self.global_step,
                                                   FLAGS.decay_steps,
                                                   FLAGS.decay_rate,
                                                   staircase=True)

        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lrn_rate,
        #                                            momentum=FLAGS.momentum).minimize(self.cost,
        #                                                                              global_step=self.global_step)
        # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lrn_rate,
        #                                             momentum=FLAGS.momentum,
        #                                             use_nesterov=True).minimize(self.cost,
        #                                                                         global_step=self.global_step)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate,
                                                beta1=FLAGS.beta1,
                                                beta2=FLAGS.beta2).minimize(self.loss,
                                                                            global_step=self.global_step)
        train_ops = [self.optimizer] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len,merge_repeated=False)
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits,
                                                                    self.seq_len,
                                                                    merge_repeated=False)
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1, name="dense_decoded")

    def _conv2d(self, x, name, filter_size, in_channels, out_channels, strides):
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='DW',
                                     shape=[filter_size[0], filter_size[1], in_channels, out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())

            b = tf.get_variable(name='bais',
                                shape=[out_channels],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME')

        return tf.nn.bias_add(con2d_op, b)

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)
            # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            x_bn = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            x_bn.set_shape(x.get_shape())

            return x_bn

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _max_pool(self, x, ksize, strides):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, strides, strides, 1],
                              padding='SAME',
                              name='max_pool')

    def _spatial_transform(self, x):
        ## x shape: [N, W, H, C=1]
        identity = np.array([[1., 0., 0.],[0., 1., 0.]])
        identity = identity.flatten()
        #theta = tf.Variable(dtype=np.float32, initial_value=identity)

        conv1_loc = tf.layers.conv2d(x, 16, 3, padding='same', activation=tf.nn.relu, name='conv1_loc')
        pool1_loc = tf.layers.max_pooling2d(conv1_loc, 2, 2)
        conv2_loc = tf.layers.conv2d(pool1_loc, 32, 3, padding='same', activation=tf.nn.relu, name='conv2_loc')
        pool2_loc = tf.layers.max_pooling2d(conv2_loc, 2, 2)
        flat_loc = tf.contrib.layers.flatten(pool2_loc)
        fc1_loc = tf.contrib.layers.fully_connected(flat_loc, 256, activation_fn=tf.nn.relu, scope='fc1_loc')
        #ac1_loc = tf.nn.tanh(fc1_loc)
        fc2_loc = tf.contrib.layers.fully_connected(fc1_loc, 6, activation_fn=None, weights_initializer=tf.zeros_initializer(), biases_initializer=tf.constant_initializer(identity), scope='fc2_loc')
        #ac2_loc = tf.nn.tanh(fc2_loc)
        stn = st.transformer(x, fc2_loc, out_size=(FLAGS.image_height, FLAGS.image_width))
        #stn = st.transformer(x, ac2_loc, out_size=(FLAGS.image_height, FLAGS.image_width))
        return stn
