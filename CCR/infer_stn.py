import os
import time

import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf

from CCR import utils, cnn_lstm_ctc_ocr
from .preparedata import PrepareData

FLAGS = utils.FLAGS
import math
import argparse

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
        val_feeder, num_samples = self.input_batch_generator(self.split_name, is_training=False,
                                                             batch_size=FLAGS.batch_size)

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

                # print the decode resultEvaluateModel

            # summary_str, step = sess.run([CCR.merged_summay, CCR.global_step])
            # eval_writer.add_summary(summary_str, step)
            return

    def infer_model(self, img):

        # image processed
        img = img.astype(np.float32) / 255.
        img = cv2.resize(img, (FLAGS.image_width, FLAGS.image_height))
        img = np.reshape(img, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])

        # CCR
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
            # restore CCR finish

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
        # need to load CCR before call this function

        # image processed
        img = img.astype(np.float32) / 255.
        img = cv2.resize(img, (FLAGS.image_width, FLAGS.image_height))
        img = np.reshape(img, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])

        # start predict
        inputs = [img]
        feed = {model.inputs: inputs}
        predictions = sess.run(model.trans_1, feed)



        return predictions, img


# -------------------------------------------

from tensorflow.python.tools import inspect_checkpoint as chkp


if __name__ == "__main__":
    # config = configure.Config(root_path=root_path)
    ocr_checkpoint_path = "/data2/CNN_LSTM_CTC_Tensorflow/checkpoint"

    with tf.get_default_graph().as_default():
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            cnn_lstm_ctc = EvaluateModel()
            ocr_model = cnn_lstm_ctc_ocr.LSTMOCR('eval')
            ocr_model.build_graph()

            if tf.gfile.IsDirectory(ocr_checkpoint_path):
                checkpoint_file = tf.train.latest_checkpoint(ocr_checkpoint_path)
            else:
                checkpoint_file = ocr_checkpoint_path

            # show tensors in the checkpoint
            chkp.print_tensors_in_checkpoint_file(checkpoint_file, tensor_name='', all_tensors=False)

            # get variable to restore
            ocr_stn_scope_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stn-1')

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            ocr_restore = tf.train.Saver(ocr_stn_scope_to_restore)
            print(checkpoint_file)
            ocr_restore.restore(sess, checkpoint_file)
 
            # get images
            img_dir = '/data2/CNN_LSTM_CTC_Tensorflow/imgs/val'
            img_names_list = os.listdir(img_dir)

            for img_name in img_names_list:
                try:
                    img_path = os.path.join(img_dir, img_name)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    stn_predict, img = cnn_lstm_ctc.marge_infer_model(sess, ocr_model, img)

                    print("img path: %s, img size: (%d, %d), " % (img_path, img.shape[0], img.shape[1]), end='')


                    #Saving plot
                    figc=np.concatenate((stn_predict[0],img),axis=1)
                    plt.imsave('./imgs/stn_output4/'+img_name+'_stn.png', stn_predict[0])
                    #img.save(os.path.join("./imgs/stn_output/",img_name+"_stn.png"))
                except Exception as e:
                    print("--------------")
                    print(e)
                    print(img_name)
                    print("--------------")
