# configuration for gunicorn
# use `gunicorn -c PATH or python:MODULE_NAME` to import config
# e.g. `gunicorn manage:app -c gc_config.py`

bind = ":5001"
workers = 4
worker_class = "gevent"
errorlog = '-'
loglevel = "info"