import requests
from flask import current_app as app
from celery.utils.log import get_task_logger
import rapidjson as json

from .. import celery
from .utilities import ccr_decode, image_web_save_encode


logger = get_task_logger(__name__)


@celery.task()
def ccr_task(b64_images):
    logger.info("CCR Task Start")
    batch_size = len(b64_images)
    body = {
        "signature_name": "ccr",
        "instances": []
    }
    for b64_image in b64_images:
        body["instances"].append(
            {
                "image_bytes": {"b64": b64_image}
            }
        )
    # logger.info(json.dumps(body))
    logger.info(app.config["CCR_URL"])
    headers = {"content-type": "application/json"}
    resp = requests.post(url=app.config["CCR_URL"], json=body, headers=headers)
    # logger.info(resp)
    result = resp.json()

    # CCR decode
    txt_list = list()
    for idx in range(batch_size):
        # response_ccr_str = [resp["predictions"][idx]['ccr_dense_decoded']]
        # # flag(like ccr_dense_decoded) will be ignore, if there are only one output of tf serving
        response_ccr_str = [result["predictions"][idx]]
        txt = ccr_decode(response_ccr_str)
        txt_list.append(txt)

    return txt_list
