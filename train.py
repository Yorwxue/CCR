import datetime
import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf

import cnn_lstm_ctc_ocr
import utils
import helper
from preparedata import PrepareData
FLAGS = utils.FLAGS
import math

logger = logging.getLogger('Traing for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)

data_prep = PrepareData()


def train(mode='train'):
    model = cnn_lstm_ctc_ocr.LSTMOCR(mode, batch_size=FLAGS.batch_size)
    model.build_graph()

    print('loading train data, please wait---------------------')
    train_feeder, num_train_samples = data_prep.input_batch_generator('train', batch_size=FLAGS.batch_size, data_dir=FLAGS.data_dir)
    print('get image: ', num_train_samples)

    print('loading validation data, please wait---------------------')
    val_feeder, num_val_samples = data_prep.input_batch_generator('val', batch_size=FLAGS.batch_size, data_dir=FLAGS.data_dir)
    print('get image: ', num_val_samples)

    num_batches_per_epoch = int(math.ceil(num_train_samples / float(FLAGS.batch_size)))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        accuracy = 0.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                # the global_step will restore sa well
                saver.restore(sess, ckpt)
                print('restore from the checkpoint{0}'.format(ckpt))

        print('=============================begin training=============================')
        for cur_epoch in range(FLAGS.num_epochs):
            start_time = time.time()
            batch_time = time.time()

            # the tracing part
            for cur_batch in range(num_batches_per_epoch):
                if (cur_batch + 1) % 100 == 0:
                    print('batch', cur_batch, ': time', time.time() - batch_time)
                batch_time = time.time()
                batch_inputs, batch_labels, _ = next(train_feeder)
                # batch_inputs,batch_seq_len,batch_labels=utils.gen_batch(FLAGS.batch_size)
                feed = {model.inputs: batch_inputs,
                        model.labels: batch_labels}

                # Note: qrnn need to know the batch size
                if len(batch_inputs) != FLAGS.batch_size:
                    print("rest %d data, drop and continue the next batch" % (len(batch_inputs)))
                    continue

                # if summary is needed
                # batch_cost,step,train_summary,_ = sess.run([cost,global_step,merged_summay,optimizer],feed)

                #print("----------------------------")
                #print(sess.run([stn_output], feed))
                #print("----------------------------")
                #exit()

                if accuracy < 0.3:
                    summary_str, batch_cost, step, _ = \
                        sess.run([model.merged_summay, model.cost, model.global_step,
                                  model.train_op_0], feed)
                else:
                    summary_str, batch_cost, step, _ = \
                        sess.run([model.merged_summay, model.cost, model.global_step,
                                  model.train_op], feed)
                # calculate the cost

                train_writer.add_summary(summary_str, step)

                # save the checkpoint
                if step % FLAGS.save_steps == 1:
                    if not os.path.isdir(FLAGS.checkpoint_dir):
                        os.mkdir(FLAGS.checkpoint_dir)
                    logger.info('save the checkpoint of{0}', format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'),
                               global_step=step)

                # train_err += the_err * FLAGS.batch_size
                # do validation
                if step % FLAGS.validation_steps == 0:
                    val_inputs, val_labels, ori_labels = next(val_feeder)

                    while len(val_inputs) != FLAGS.batch_size:
                        print("rest %d validation data, drop and continue the next batch" % (len(val_inputs)))
                        val_inputs, val_labels, ori_labels = next(val_feeder)

                    val_feed = {model.inputs: val_inputs,
                                model.labels: val_labels}

                    dense_decoded, lr = \
                        sess.run([model.dense_decoded, model.lrn_rate],
                                 val_feed)

                    # print the decode result
                    accuracy = utils.accuracy_calculation(ori_labels, dense_decoded,
                                                     ignore_value=-1, isPrint=True)

                    # train_err /= num_train_samples
                    now = datetime.datetime.now()
                    log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                          "accuracy = {:.5f},train_cost = {:.5f}, " \
                          ", time = {:.3f},lr={:.8f}"
                    print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                     cur_epoch + 1, FLAGS.num_epochs, accuracy, batch_cost,
                                     time.time() - start_time, lr))


def main(_):
    if FLAGS.mode == 'train':
        train(FLAGS.mode)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
