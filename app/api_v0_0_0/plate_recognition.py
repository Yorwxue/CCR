from flask import request, jsonify
from flask import current_app as app
from celery.contrib.abortable import AbortableAsyncResult
from celery.utils.log import get_task_logger
from time import sleep

from . import api_v0_0_0 as api
from . import errors, tasks
from .. import celery


logger = get_task_logger(__name__)


@api.route("/plate-recognition/CCR", methods=["POST"], endpoint="plate_recognition/CCR")
def ccr():
    req_data = request.get_json()
    # print("Received Data: ", req_data)
    keys = ["b64_images"]
    if set(keys) == set(req_data.keys()):
        tasks.ccr_task.delay(req_data["b64_images"])
        return jsonify({'state': 'task accepted'}), 201
    else:
        return errors.bad_request('wrong format')
