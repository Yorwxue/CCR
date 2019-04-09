import datetime
import logging
import os
import time
import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

import cnn_lstm_ctc_ocr
import utils
from preparedata import PrepareData

import math
import argparse

FLAGS = utils.FLAGS

# log_dir = './log/infers'
logger = logging.getLogger('Traing for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)


def image_reader(img_path):
    input_image = misc.imread(img_path)
    if input_image.shape[2] > 3:
        input_image = input_image[:, :, :3]
    return input_image


class EvaluateModel(PrepareData):
    def __init__(self):
        PrepareData.__init__(self)
        return

    def parse_param(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--split_name', help='which split of dataset to use', default="eval")
        parser.add_argument('-c', '--checkpoint_path', help='which checkpoint to use', default="./checkpoint/")
        args = parser.parse_args()

        self.checkpoint_path = args.checkpoint_path
        self.split_name = args.split_name

        return

    def eval_model(self):
        model = cnn_lstm_ctc_ocr.LSTMOCR('eval')
        model.build_graph()
        val_feeder, num_samples = self.input_batch_generator(self.split_name,
                                                             batch_size=FLAGS.batch_size,
                                                             data_dir=FLAGS.data_dir)

        num_batches_per_epoch = int(math.ceil(num_samples / float(FLAGS.batch_size)))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            # eval_writer = tf.summary.FileWriter("{}/{}".format(log_dir, self.split_name), sess.graph)

            if tf.gfile.IsDirectory(self.checkpoint_path):
                checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_path)
            else:
                checkpoint_file = self.checkpoint_path
            print('Evaluating checkpoint_path={}, split={}, num_samples={}'.format(checkpoint_file, self.split_name,
                                                                                   num_samples))

            saver.restore(sess, checkpoint_file)
            true = 0.
            false = 0.
            for i in range(num_batches_per_epoch):
                inputs, labels, _ = next(val_feeder)
                feed = {model.inputs: inputs,
                        model.labels: labels}
                start = time.time()
                _, predictions = sess.run([model.names_to_updates, model.dense_decoded], feed)
                # --
                gt_encode = self.label_from_sparse_tuple(labels)
                gt = list()
                pred = list()
                for j in range(len(gt_encode)):
                    gt_code = [utils.decode_maps[c] if c != -1 else '' for c in gt_encode[j]]
                    gt_code = ''.join(gt_code)
                    gt.append(gt_code)
                for j in range(len(predictions)):
                    code = [utils.decode_maps[c] if c != -1 else '' for c in predictions[j]]
                    code = ''.join(code)
                    pred.append(code)
                for j in range(len(gt)):
                    print("%s  :  %s" % (gt[j], pred[j]))
                    if gt[j] == pred[j]:
                        true += 1
                    else:
                        false += 1
                # --
                elapsed = time.time()
                elapsed = elapsed - start
                print('{}/{}, {:.5f} seconds.'.format(i, num_batches_per_epoch, elapsed))
                # print the decode result
            print("accuracy: %f" % (true/(true+false)))

            # summary_str, step = sess.run([model.merged_summay, model.global_step])
            # eval_writer.add_summary(summary_str, step)
            return

    def infer_model(self, img):

        # image processed
        img = img.astype(np.float32) / 255.
        img = cv2.resize(img, (FLAGS.image_width, FLAGS.image_height))
        img = np.reshape(img, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])

        # model
        model = cnn_lstm_ctc_ocr.LSTMOCR('eval')
        model.build_graph()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

            if tf.gfile.IsDirectory(self.checkpoint_path):
                checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_path)
            else:
                checkpoint_file = self.checkpoint_path
            print('Evaluating checkpoint_path={}'.format(checkpoint_file))

            saver.restore(sess, checkpoint_file)
            # restore model finish

            inputs = [img]
            feed = {model.inputs: inputs}
            # start = time.time()
            predictions = sess.run(model.dense_decoded, feed)

            pred = list()

            for j in range(len(predictions)):
                code = [utils.decode_maps[c] if c != -1 else '' for c in predictions[j]]
                code = ''.join(code)
                pred.append(code)
                # print("%s" % pred[-1])

            # elapsed = time.time()
            # elapsed = elapsed - start
            # print('Spent {:.5f} seconds.'.format(elapsed))
            return pred[-1]

    def marge_infer_model(self, sess, model, img):
        # need to load model before call this function

        # image processed
        img = img.astype(np.float32) / 255.
        img = cv2.resize(img, (FLAGS.image_width, FLAGS.image_height))
        img = np.reshape(img, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])

        # start predict
        inputs = [img]
        feed = {model.inputs: inputs}
        predictions = sess.run(model.dense_decoded, feed)

        pred = list()

        for j in range(len(predictions)):
            code = [utils.decode_maps[c] if c != -1 else '' for c in predictions[j]]
            code = ''.join(code)
            pred.append(code)

        return pred[-1]


if __name__ == "__main__":
    """
    img_path = "../testing/3H9255.jpg"

    input_image = image_reader(img_path)

    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:
            cnn_lstm_ctc = EvaluateModel()
            ocr_model = cnn_lstm_ctc_ocr.LSTMOCR('eval')
            ocr_model.build_graph()

            if tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
                checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            else:
                checkpoint_file = FLAGS.checkpoint_dir

            ocr_cnn_scope_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cnn')
            ocr_lstm_scope_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lstm')
            ocr_stn_scope_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stn-1')
            ocr_restore = tf.train.Saver(ocr_cnn_scope_to_restore + ocr_lstm_scope_to_restore + ocr_stn_scope_to_restore)
            ocr_restore.restore(sess, checkpoint_file)

            # show tensors in the graph
            for each_layer in tf.global_variables():
                print(each_layer)

            start_time = time.time()
            # txt_ocr = cnn_lstm_ctc.marge_infer_model(sess, ocr_model, input_image)
            # print("ccr model spent %f sec" % (time.time() - start_time))
            # print(txt_ocr)
    """
    cnn_lstm_ctc = EvaluateModel()
    cnn_lstm_ctc.parse_param()
    cnn_lstm_ctc.eval_model()
