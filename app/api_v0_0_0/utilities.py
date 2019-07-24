from celery.utils.log import get_task_logger
import cv2
import pybase64 as base64
import numpy as np

import CCR.utils as utils


logger = get_task_logger(__name__)


def ccr_decode(predictions):
    pred = list()
    for j in range(len(predictions)):
        code = [utils.decode_maps[c] if c != -1 else '' for c in predictions[j]]
        code = ''.join(code)
        pred.append(code)
        # logger.info("CCR decode: %s" % code)
    return pred


def image_encode(image):
    # image -> bytes -> string
    input_bytes = cv2.imencode(".jpg", image)[1].tostring()
    input_image = base64.b64encode(input_bytes)
    image_content = input_image.decode("utf-8")
    return image_content


def image_decode(image_str):
    image_bytes = base64.b64decode(image_str)
    nparr = np.fromstring(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def image_web_save_encode(image):
    input_bytes = cv2.imencode(".jpg", image)[1].tostring()
    input_image = base64.urlsafe_b64encode(input_bytes)
    image_content = input_image.decode("utf-8")
    return image_content


def image_web_save_decode(image_str):
    image_bytes = base64.urlsafe_b64decode(image_str)
    nparr = np.fromstring(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image
