import os
import requests
import pybase64 as base64
import rapidjson as json
import cv2

from config import config
from app.api_v0_0_0.utilities import ccr_decode
from CCR.polyCrop import ratioImputation
from app.api_v0_0_0.utilities import image_encode

configure = config["default"]


def ccr(images, URL,
        headers={"content-type": "application/json"}):
    batch_size = len(images)
    # ccr_body = {
    #     "signature_name": "ccr",
    #     # "instances": [{"image_bytes": {"b64": image_content}}]
    #     "inputs": {"image_bytes": [{"b64": image_content}]}
    # }
    ccr_body = {
        "signature_name": "ccr",
        "instances": [
            # ex:
            # {"image_bytes": {"b64": image_content}}
        ]
    }
    # print('ccr encode')
    # now=time.time()
    # for i in images:
    #     ccr_body["instances"].append(image_encode(i))
    # with multiprocessing.Pool(1) as p:
    #     ccr_body["instances"] += p.map(image_encode, images)

    for i in images:
        ccr_body["instances"].append(
            {"image_bytes": {"b64": image_encode(i)}}
        )
    # with multiprocessing.Pool(1) as p:
    #     ccr_body["instances"] += p.map(image_encode, images)
    # cost(now)
    #
    # now = time.time()
    r = requests.post(URL, data=json.dumps(ccr_body), headers=headers)
    # print(r.text)
    resp = json.loads(r.text)
    # response_string = resp["outputs"]['ret_img_str_bytes']["b64"]
    # response_image_bytes = base64.b64decode(response_string)
    # cost(now)

    # ccr decode
    txt_list = list()
    for idx in range(batch_size):
        # response_ccr_str = [resp["predictions"][idx]['ccr_dense_decoded']]
        # # flag(like ccr_dense_decoded) will be ignore, if there are only one output of tf serving
        response_ccr_str = [resp["predictions"][idx]]
        txt = ccr_decode(response_ccr_str)
        txt_list.append(txt)

    return txt_list


if __name__ == "__main__":

    # ccr
    # """
    image_path = r"/mnt/hdd1/SUCK/dataset/123.jpg"
    image = cv2.imread(image_path)
    ratio_img = ratioImputation(image, target_ration=(60, 180))
    txt_list = ccr([ratio_img], URL=configure.CCR_URL)
    for txt in txt_list:
        print(txt)
    # """
