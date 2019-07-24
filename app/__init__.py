from celery import Celery
from flask import Flask

from config import config, Config


celery = Celery(__name__, broker=Config.BROKER_URL)


def create_app(config_name):
    app = Flask(__name__, instance_relative_config=True)

    # config
    app.config.from_object(config[config_name])
    app.config.from_pyfile("/mnt/hdd1/flask_experiment/config.py")

    # blueprint
    from .views import views
    app.register_blueprint(views)

    from .api_v0_0_0 import api_v0_0_0
    app.register_blueprint(api_v0_0_0, url_prefix='/api/v0.0.0')

    # extensions
    celery.conf.update(app.config)

    return app
