from flask import Blueprint

api_v0_0_0 = Blueprint('api_v0_0_0', __name__)

from . import errors, plate_recognition
