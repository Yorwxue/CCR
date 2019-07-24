from flask import jsonify, abort, request
from . import api_v0_0_0 as api


@api.errorhandler(400)
def bad_request(message):
    if request.accept_mimetypes.accept_json:
        response = jsonify({'error': 'bad request', 'message': message})
        response.status_code = 400
        return response
    return abort(400)


@api.errorhandler(401)
def unauthorized(message):
    if request.accept_mimetypes.accept_json:
        response = jsonify({'error': 'unauthorized', 'message': message})
        response.status_code = 401
        return response
    return abort(401)


@api.errorhandler(403)
def forbidden(message):
    if request.accept_mimetypes.accept_json:
        response = jsonify({'error': 'forbidden', 'message': message})
        response.status_code = 403
        return response
    return abort(403)


@api.errorhandler(404)
def forbidden(message):
    if request.accept_mimetypes.accept_json:
        response = jsonify({'error': 'not found', 'message': message})
        response.status_code = 404
        return response
    return abort(404)


@api.errorhandler(405)
def forbidden(message):
    if request.accept_mimetypes.accept_json:
        response = jsonify({'error': 'method not allowed', 'message': message})
        response.status_code = 405
        return response
    return abort(405)