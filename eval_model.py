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

try:
    import configure.Config as config_obj
except:
    class config_obj(object):
        def __init__(self, root_path):
            self.root_path = root_path

            # ocr config
            self.ocr_checkpoint_path = os.path.join(self.root_path, "checkpoint")

            if not os.path.exists(self.ocr_checkpoint_path):
                os.makedirs(self.ocr_checkpoint_path)

            self.batch_size = 64
            self.log_dir = os.path.join(self.root_path, "log")

            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                os.makedirs(os.path.join(self.log_dir, 'train'))

            self.num_epochs = 500
            self.image_height = 60
            self.image_width = 180
            self.image_channel = 3
            self.restore = False
            self.save_steps = 1000
            self.validation_steps = 100


# log_dir = './log/infers'
logger = logging.getLogger('Traing for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)


def image_reader(img_path):
    input_image = misc.imread(img_path)
    if input_image.shape[2] > 3:
        input_image = input_image[:, :, :3]
    return input_image


class EvaluateModel(PrepareData):
    def __init__(self, config):
        self.config = config
        PrepareData.__init__(self)
        return

    def train(self, train_dir=None, val_dir=None, mode='train'):
        model = cnn_lstm_ctc_ocr.LSTMOCR(mode)
        model.build_graph()

        print('loading train data')
        train_feeder = utils.DataIterator(data_dir=train_dir)
        print('size: ', train_feeder.size)

        print('loading validation data')
        val_feeder = utils.DataIterator(data_dir=val_dir)
        print('size: {}\n'.format(val_feeder.size))

        num_train_samples = train_feeder.size  # 100000
        num_batches_per_epoch = int(num_train_samples / self.config.batch_size)  # example: 100000/100

        num_val_samples = val_feeder.size
        num_batches_per_epoch_val = int(num_val_samples / self.config.batch_size)  # example: 10000/100
        shuffle_idx_val = np.random.permutation(num_val_samples)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            train_writer = tf.summary.FileWriter(os.path.join(self.config.log_dir, 'train'), sess.graph)
            if self.config.restore:
                ckpt = tf.train.latest_checkpoint(self.config.ocr_checkpoint_path)
                if ckpt:
                    # the global_step will restore sa well
                    saver.restore(sess, ckpt)
                    print('restore from checkpoint{0}'.format(ckpt))

            print('=============================begin training=============================')
            for cur_epoch in range(self.config.num_epochs):
                shuffle_idx = np.random.permutation(num_train_samples)
                train_cost = 0
                start_time = time.time()
                batch_time = time.time()

                # the training part
                for cur_batch in range(num_batches_per_epoch):
                    if (cur_batch + 1) % 100 == 0:
                        print('batch', cur_batch, ': time', time.time() - batch_time)
                    batch_time = time.time()
                    indexs = [shuffle_idx[i % num_train_samples] for i in
                              range(cur_batch * self.config.batch_size, (cur_batch + 1) * self.config.batch_size)]
                    batch_inputs, _, batch_labels = \
                        train_feeder.input_index_generate_batch(indexs)
                    # batch_inputs,batch_seq_len,batch_labels=utils.gen_batch(self.config.batch_size)
                    feed = {model.inputs: batch_inputs,
                            model.labels: batch_labels}

                    # if summary is needed
                    summary_str, batch_cost, step, _ = \
                        sess.run([model.merged_summay, model.cost, model.global_step, model.train_op], feed)
                    # calculate the cost
                    train_cost += batch_cost * self.config.batch_size

                    train_writer.add_summary(summary_str, step)

                    # save the checkpoint
                    if step % self.config.save_steps == 1:
                        if not os.path.isdir(self.config.ocr_checkpoint_path):
                            os.mkdir(self.config.ocr_checkpoint_path)
                        logger.info('save checkpoint at step {0}', format(step))
                        saver.save(sess, os.path.join(self.config.ocr_checkpoint_path, 'ocr-model'), global_step=step)

                    # train_err += the_err * self.config.batch_size
                    # do validation
                    if step % self.config.validation_steps == 0:
                        acc_batch_total = 0
                        lastbatch_err = 0
                        lr = 0
                        for j in range(num_batches_per_epoch_val):
                            indexs_val = [shuffle_idx_val[i % num_val_samples] for i in
                                          range(j * self.config.batch_size, (j + 1) * self.config.batch_size)]
                            val_inputs, _, val_labels = \
                                val_feeder.input_index_generate_batch(indexs_val)
                            val_feed = {model.inputs: val_inputs,
                                        model.labels: val_labels}

                            dense_decoded, lastbatch_err, lr = \
                                sess.run([model.dense_decoded, model.cost, model.lrn_rate],
                                         val_feed)

                            # print the decode result
                            ori_labels = val_feeder.the_label(indexs_val)
                            acc = utils.accuracy_calculation(ori_labels, dense_decoded,
                                                             ignore_value=-1, isPrint=True)
                            acc_batch_total += acc

                        accuracy = (acc_batch_total * self.config.batch_size) / num_val_samples

                        avg_train_cost = train_cost / ((cur_batch + 1) * self.config.batch_size)

                        # train_err /= num_train_samples
                        now = datetime.datetime.now()
                        log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                              "accuracy = {:.3f},avg_train_cost = {:.3f}, " \
                              "lastbatch_err = {:.3f}, time = {:.3f},lr={:.8f}"
                        print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                         cur_epoch + 1, self.config.num_epochs, accuracy, avg_train_cost,
                                         lastbatch_err, time.time() - start_time, lr))

    def parse_param(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--split_name', help='which split of dataset to use', default="eval")
        parser.add_argument('-c', '--checkpoint_path', help='which checkpoint to use', default="./checkpoint_fin/")
        args = parser.parse_args()

        self.checkpoint_path = args.checkpoint_path
        self.split_name = args.split_name

        return

    def eval_model(self):
        model = cnn_lstm_ctc_ocr.LSTMOCR('eval')
        model.build_graph()
        val_feeder, num_samples = self.input_batch_generator(self.split_name, is_training=False,
                                                             batch_size=self.config.batch_size)

        num_batches_per_epoch = int(math.ceil(num_samples / float(self.config.batch_size)))

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
                # --
                elapsed = time.time()
                elapsed = elapsed - start
                print('{}/{}, {:.5f} seconds.'.format(i, num_batches_per_epoch, elapsed))

                # print the decode result

            # summary_str, step = sess.run([model.merged_summay, model.global_step])
            # eval_writer.add_summary(summary_str, step)
            return

    def infer_model(self, img):

        # image processed
        img = img.astype(np.float32) / 255.
        img = cv2.resize(img, (self.config.image_width, self.config.image_height))
        img = np.reshape(img, [self.config.image_height, self.config.image_width, self.config.image_channel])

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
        img = cv2.resize(img, (self.config.image_width, self.config.image_height))
        img = np.reshape(img, [self.config.image_height, self.config.image_width, self.config.image_channel])

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
    img_path = "../testing/3H9255.jpg"

    input_image = image_reader(img_path)

    config = config_obj(root_path=os.path.abspath(os.path.join(os.getcwd())))

    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:
            cnn_lstm_ctc = EvaluateModel(config)
            ocr_model = cnn_lstm_ctc_ocr.LSTMOCR('eval')
            ocr_model.build_graph()

            if tf.gfile.IsDirectory(config.ocr_checkpoint_path):
                checkpoint_file = tf.train.latest_checkpoint(config.ocr_checkpoint_path)
            else:
                checkpoint_file = config.ocr_checkpoint_path

            ocr_cnn_scope_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cnn')
            ocr_lstm_scope_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lstm')
            ocr_stn_scope_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stn-1')
            ocr_restore = tf.train.Saver(ocr_cnn_scope_to_restore + ocr_lstm_scope_to_restore + ocr_stn_scope_to_restore)
            ocr_restore.restore(sess, checkpoint_file)

            # show tensors in the graph
            for each_layer in tf.global_variables():
                print(each_layer)

            start_time = time.time()
            txt_ocr = cnn_lstm_ctc.marge_infer_model(sess, ocr_model, input_image)
            print("ccr model spent %f sec" % (time.time() - start_time))
            print(txt_ocr)