"""
optical character detection based on EAST
"""
import os
import numpy as np
import tensorflow as tf
from east import lanms
import time
import east.east_model as east_model
import cv2
# import pytesseract
from PIL import Image
import Levenshtein
# import xml.etree.ElementTree as ET

from east.icdar import restore_rectangle
from east.polyCrop import polyCrop, ratioImputation

from eval_model import EvaluateModel


class OCD(object):
    """
    image format: opencv
    Optical Character Detection
    """

    def __init__(self, checkpoint_path='pretrain_model/east_icdar2015_resnet_v1_50_rbox/', gpu_list=0):
        self.gpu_list = gpu_list

        self.graph = tf.get_default_graph().as_default()

        self.input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        self.f_score, self.f_geometry = east_model.model(self.input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.as_default()

        ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
        model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
        print('Restore from {}'.format(model_path))
        saver.restore(self.sess, model_path)

        # used in eval
        self.IoU_threshold = 0.8

    def detection(self, image):
        char_img_list = list()
        # with self.graph:
        # with self.sess:
        marked_img = np.copy(image)
        im = marked_img[:, :, ::-1]

        start_time = time.time()
        im_resized, (ratio_h, ratio_w) = self.__resize_image(im)

        timer = {'net': 0, 'restore': 0, 'nms': 0}
        start = time.time()
        score, geometry = self.sess.run([self.f_score, self.f_geometry],
                                        feed_dict={self.input_images: [im_resized]})
        timer['net'] = time.time() - start

        boxes, timer = self.__detect(score_map=score, geo_map=geometry, timer=timer)

        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        # print('[timing] {}'.format(duration))

        if boxes is not None:
            for box in boxes:
                # to avoid submitting errors
                box = self.__sort_poly(box.astype(np.int32))

                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue

                # crop image
                # ----
                max_x = np.max(np.asarray(box)[:, 0])
                min_x = np.min(np.asarray(box)[:, 0])
                max_y = np.max(np.asarray(box)[:, 1])
                min_y = np.min(np.asarray(box)[:, 1])
                point_rect = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

                display_img, masked_image = polyCrop(image, rect_box=point_rect, poly_box=box)
                # ratio_img = ratioImputation(masked_image, target_ration=(60, 180))
                # ---

                # draw boxe
                cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                              color=(255, 255, 0), thickness=1)

                char_img_list.append(masked_image)
                marked_img = im[:, :, ::-1]

        return char_img_list, marked_img, boxes

    def __detect(self, score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
        """
        restore text boxes from score map and geo map
        :param score_map:
        :param geo_map:
        :param timer:
        :param score_map_thresh: threshhold for score map
        :param box_thresh: threshhold for boxes
        :param nms_thres: threshold for nms
        :return:
        """
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, ]
        # filter the score map
        xy_text = np.argwhere(score_map > score_map_thresh)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        # restore
        start = time.time()
        text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
        # print('{} text boxes before nms'.format(text_box_restored.shape[0]))
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        timer['restore'] = time.time() - start
        # nms part
        start = time.time()
        # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
        boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
        timer['nms'] = time.time() - start

        if boxes.shape[0] == 0:
            return None, timer

        # here we filter some low score boxes by the average score map, this is different from the orginal paper
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > box_thresh]

        return boxes, timer

    def __sort_poly(self, p):
        min_axis = np.argmin(np.sum(p, axis=1))
        p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]

    def __resize_image(self, im, max_side_len=2400):
        """
        resize image to a size multiple of 32 which is required by the network
        :param im: the resized image
        :param max_side_len: limit of max image size to avoid out of memory in gpu
        :return: the resized image and the resize ratio
        """
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # limit the max side
        if max(resize_h, resize_w) > max_side_len:
            ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
        else:
            ratio = 1.
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
        resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
        im = cv2.resize(im, (int(resize_w), int(resize_h)))

        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return im, (ratio_h, ratio_w)

    def minimum_external_rectangular(self, boxes_list):
        """
        using lines horizontal to axis to represent the bounding box.
        lines are represented by x1, x2, y1, y2

          0_____x1_______x2_______
          |     |        |
        y1|-----O--------O--------
          |     |////////|
          |     |////////|
        y2|-----O--------O--------
          |     |        |

        :param boxes_list:
        :return:
        """
        new_boxes_list = list()
        for box in boxes_list:
            new_boxes_list.append([
                min([box[0][0], box[3][0]]),  # left
                min([box[0][1], box[1][1]]),  # upper
                max([box[1][0], box[2][0]]),  # right
                max([box[2][1], box[3][1]])  # bottom
            ])
        return new_boxes_list

    def eval(self, data_paths, labels):
        """
        compute accuracy by region of union
        """
        recall_true = 0.
        recall_false = 0.
        prec_true = 0.
        prec_false = 0.

        # check dataset
        if len(data_paths) != len(labels):
            print("dataset error")
            return -1

        # OCD
        for data_idx, data_path in enumerate(data_paths):
            image_path = data_path
            try:
                image = cv2.imread(image_path)
                W, H, _ = image.shape
            except:
                print("load image error: ", data_path)
                continue

            if len(labels[data_idx]):
                try:
                    # east box format:
                    #  0------1
                    #  |      |
                    #  |      |
                    #  3------2
                    _, _, detect_boxes = self.detection(image)
                except Exception as e:
                    print(e)
                    print(image_path)
                    continue

                # if type(detect_boxes) == type(None):  # OCD error: return a NoneType object
                if isinstance(detect_boxes, type(None)):  # OCD error: return a NoneType object
                    # recall
                    recall_false += len(labels[data_idx])
                    continue

                if len(detect_boxes) == 0:  # not text box found in this image
                    # recall
                    recall_false += len(labels[data_idx])
                    continue

                # show result
                """
                # sorted boxes
                detect_boxes = sorted(detect_boxes, key=lambda box: (box[0][0] + box[0][1]))
                labels[data_idx] = sorted(labels[data_idx], key=lambda box: (box[0][0] + box[0][1]))

                # draw boxe

                # ----------
                for box_idx, box in enumerate(detect_boxes):
                    box = self.__sort_poly(box.astype(np.int32))
                    cv2.polylines(image, [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                  color=(255, 255, 0), thickness=1)
                    cv2.putText(image, str(box_idx), (box[0][0], box[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2, cv2.LINE_AA)

                for box_idx, box in enumerate(labels[data_idx]):
                    box = np.asarray(box).astype(np.int32)
                    cv2.polylines(image, [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                  color=(0, 0, 0), thickness=1)
                    cv2.putText(image, str(box_idx), (box[0][0], box[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
                # ----------

                # cv2 to pil, and show
                # --
                cv2_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im)
                img = pil_im.convert("RGB")
                img.show()
                # --
                # """

                # change format for IoU computing
                # (1) ground true label
                label_new_boxes = self.minimum_external_rectangular(labels[data_idx])

                # (2) east label
                detect_new_boxes = self.minimum_external_rectangular(detect_boxes)

                # IoU array:
                #               prediction
                #             o------------o
                #             |            |
                # ground true |            |
                #             |            |
                #             o------------o
                IoU_array = np.zeros((len(label_new_boxes), len(detect_new_boxes)))
                for true_box_idx in range(len(label_new_boxes)):
                    for pred_box_idx in range(len(detect_new_boxes)):
                        IoU = self.bb_intersection_over_union(label_new_boxes[true_box_idx],
                                                              detect_new_boxes[pred_box_idx])
                        IoU_array[true_box_idx, pred_box_idx] = IoU

                # recall
                for i in range(len(label_new_boxes)):
                    if max(IoU_array[i, :]) >= self.IoU_threshold:
                        recall_true += 1
                    else:
                        recall_false += 1

                # precision
                for i in range(len(detect_new_boxes)):
                    if max(IoU_array[:, i]) >= self.IoU_threshold:
                        prec_true += 1
                    else:
                        prec_false += 1

            else:  # there isn't any boxes in this image
                continue

        recall = recall_true / (recall_true + recall_false)
        precision = prec_true / (prec_true + prec_false)
        f1_score = 2 * precision * recall / (precision + recall)

        return precision, recall, f1_score

    def bb_intersection_over_union(self, boxA, boxB):
        """
        computer Intersection Over Union(IOU)
        more detai can be find in the following url:
            https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        :param boxA:
        :param boxB:
        :return:
        """
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])  # left
        yA = max(boxA[1], boxB[1])  # upper
        xB = min(boxA[2], boxB[2])  # right
        yB = min(boxA[3], boxB[3])  # bottom

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou


# class OCR_tess(object):
#     """
#     Optical Character Recognition(tesseract)
#     --list-langs:
#         chi_sim
#         chi_tra
#         eng
#         osd
#     """
#
#     def __init__(self):
#         pass
#
#     def recognize(self, image, lang=None):
#         text = pytesseract.image_to_string(image, lang=lang)
#         return text


class OCR(object):
    """
    image format: PIL
    """
    def __init__(self, DETECTANGLE=True, leftAdjust=True, rightAdjust=True, alph=0.2, ifadjustDegree=False):
        # ocr parameter
        self.cnn_lstm_ctc = EvaluateModel()
        self.cnn_lstm_ctc.parse_param()

    def recognize(self, imgs):
        f = 1.0
        # results_list = crnnRec(np.array(img), newBox, self.leftAdjust, self.rightAdjust, self.alph, 1.0 / f, k=k)
        results_list = self.cnn_lstm_ctc.infer_model(imgs)
        return results_list

    def minimum_external_rectangular(self, boxes_list):
        """
        using lines horizontal to axis to represent the bounding box.
        lines are represented by x1, x2, y1, y2

          0_____x1_______x2_______
          |     |        |
        y1|-----O--------O--------
          |     |////////|
          |     |////////|
        y2|-----O--------O--------
          |     |        |

        :param boxes_list:
        :return:
        """
        new_boxes_list = list()
        for box in boxes_list:
            new_boxes_list.append([
                min([box[0][0], box[3][0]]),  # left
                min([box[0][1], box[1][1]]),  # upper
                max([box[1][0], box[2][0]]),  # right
                max([box[2][1], box[3][1]])  # bottom
            ])
        return new_boxes_list

    def eval(self, data_paths, labels, positions_list=None):
        """
        compute accuracy by region of union
        """

        total = 0.
        true = 0.
        I, R, D, E = 0., 0., 0., 0.

        # check dataset
        # -------------
        if len(data_paths) != len(labels):
            print("dataset error")
            return -1
        if isinstance(positions_list, list):
            if len(data_paths) != len(positions_list):
                print("dataset error")
                exit()
        # -------------

        for idx in range(len(data_paths)):
            review_flag = 0
            image_list = list()
            image = Image.open(data_paths[idx])

            # need to crop image by position
            if isinstance(positions_list, list):
                positions = self.minimum_external_rectangular(positions_list[idx])
                for box_idx in range(len(positions)):
                    image_list.append(
                        image.crop((
                            int(positions[box_idx][0]), int(positions[box_idx][1]),
                            int(positions[box_idx][2]), int(positions[box_idx][3])
                        ))
                    )  # left, upper, right, and lower pixel coordinate.
            else:
                image_list.append(image)
                labels[idx] = [labels[idx]]

            # OCR
            result_list = list()
            for img in image_list:
                total += 1
                W, H = img.size
                result_list = self.recognize(img, np.asarray([[[0, 0], [W, 0], [W, H], [0, H]]]))

                # get the top one candidate
                result = result_list[0]

                if len(result) == 0:
                    result_list.append('')
                    continue

                result = result['text']
                result_list.append(result)

            for result_idx, result in enumerate(result_list):
                # full match
                if result == labels[idx][result_idx]:
                    true += 1

                # character error rate, CER
                step_list = Levenshtein.opcodes(result.lower(), labels[idx][result_idx].lower())  # pred -> labels[idx]
                # opcodes: return tuple of 5 elements, first means operator, second to fifth means positions of start and end

                for step in step_list:
                    if step[0] == "insert":
                        I += step[4] - step[3]
                        if (step[4] - step[3]) != 0:
                            review_flag = 1
                    elif step[0] == "replace":
                        R += step[2] - step[1]
                        if (step[2] - step[1]) != 0:
                            review_flag = 1
                    elif step[0] == "delete":
                        D += step[2] - step[1]
                        if (step[2] - step[1]) != 0:
                            review_flag = 1
                    elif step[0] == "equal":
                        E += step[2] - step[1]
            if review_flag:
                pass

        print("match accuracy = %f" % (true / total))
        print("CER = %f" % ((R + D + I)/(R + D + E)))
        pass


if __name__ == '__main__':
    OCD = OCD("/home/c11tch/workspace/PycharmProjects/JC_Demo/ocr_module/EAST/pretrained_model/east_mixed_149482/")
    # OCD = OCD("/data2/relabelled/east_icdar2015_resnet_v1_50_rbox")
    OCR = OCR()

    # image_path = "/home/c11tch/workspace/PycharmProjects/EAST/data/img_1001.jpg"
    # image_path = "/home/c11tch/Pictures/123.png"
    image_path = "/home/c11tch/Pictures/aPICT0034.JPG"
    # image_path = "/data1/Dataset/OCR/chinese_board/000026.jpg"
    # image_path = "/data2/EAST_relabelled/1.jpg"

    image = cv2.imread(image_path)
    # image = Image.open(image_path)

    # EAST
    image_list, masked_image, boxes = OCD.detection(image)

    # text_list = list()

    # tesseract ocr
    """
    OCR_tess = OCR_tess()
    for idx, img in enumerate(image_list):
        char_txt = OCR_tess.recognize(img, lang="chi_sim")
        text_list.append(char_txt)

        # cv2 to pil
        cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)

        # show
        pil_im.show()

        print("text: %s" % char_txt)
    
    # cv2 to pil, and show
    cv2_im = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    pil_im.show()
    # """

    # chineseocr
    # """

    # cv2 to pil
    cv2_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    img = pil_im.convert("RGB")

    top_k = 3

    # ocr
    results_list = OCR.recognize(img, boxes, k=top_k)

    # cv2 to pil, and show
    cv2_im = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    pil_im.show()

    for idx, elem in enumerate(results_list):
        for i in range(top_k):
            print("text: %s" % elem[top_k]['text'])

        # cv2 to pil
        cv2_im = cv2.cvtColor(image_list[idx], cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        pil_im.show()
        pass
    # """

    # ---------------------------------
    # dataset = icdar2003(OCD_dataset_path="/data1/Dataset/OCR/icdar2003/Robust Reading and Text Locating/SceneTrialTest",
    #                     OCR_dataset_path="/data1/Dataset/OCR/icdar2003/Robust Word Recognition/1")

    # OCR eval
    """
    data_paths, labels = dataset.OCR_dataset()
    OCR.eval(data_paths, labels)
    # """

    # OCD eval
    """
    data_paths, labels = dataset.OCD_dataset()
    precision, recall, f1_score = OCD.eval(data_paths, labels)
    print("threshold: %.2f" % OCD.IoU_threshold)
    print("precision: %f" % precision)
    print("recall: %f" % recall)
    print("f1 score: %f" % f1_score)

    OCD.sess.close()
    # """

    # end-to-end eval
    """
    data_paths, labels, box_positions = dataset.end_to_end_dataset()

    # get crop by ground truth position
    OCR.eval(data_paths, labels, box_positions)
    # """
