class Config(object):
    DEBUG = False
    TESTING = False

    # celery
    BROKER_URL = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND = "redis://localhost:6379/1"
    CELERY_TASK_SERIALIZER = "json"


class DevelopmentConfig(Config):
    DEBUG = True

    CCR_URL = "http://10.244.0.113:8501/v1/models/ccr:predict"


config = {
    'development': DevelopmentConfig,

    # defult config
    'default': DevelopmentConfig
}
