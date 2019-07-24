import os


class Config(object):
    DEBUG = False
    TESTING = False

    # celery
    BROKER_URL = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND = "redis://localhost:6379/1"
    CELERY_TASK_SERIALIZER = "json"


class DevelopmentConfig(Config):
    DEBUG = True

    # service
    # CCR_URL = "http://10.244.0.113:8501/v1/models/ccr:predict"
    CCR_URL = "http://localhost:8501/v1/models/ccr:predict"

    # model parameter
    input_size_h = 60
    input_size_w = 180
    ccr_checkpoint_path = os.path.abspath(os.path.join(__file__, "..", "CCR", "checkpoint"))
    ccr_export_dir = os.path.abspath(os.path.join(__file__, "..", "tf_serving", "export_model", "ccr"))


config = {
    'development': DevelopmentConfig,

    # defult config
    'default': DevelopmentConfig
}
