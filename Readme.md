# Convolutional Connectionist-temporal-classification with Recurrent network(CCR)
## Deploy
### Celery
+ Celery is an asynchronous task manager.
+ Celery support following message proxy(or called "Broker""): RabbitMQ, Redis, MongoDB, Beanstalk, SQLAlchemy, Zookeeper
+ RabbitMQ is the most suggested, and second is Redis. Due to RabbitMQ is more difficult to build, so we use Redis as Broker
```bash
# Install celery with dependency package
$ pip install "celery[librabbitmq,redis,msgpack]"
```

+ You can start your task manager by the following command:
```bash
$ celery worker -A celery_worker.celery -l INFO
```

#### Redis
+ Install Redis for Celery
```bash
$ wget http://download.redis.io/redis-stable.tar.gz
$ tar xvzf redis-stable.tar.gz
$ cd redis-stable
$ make
$ sudo apt-get install tcl8.5-dev
$ make test
$ sudo apt install redis-server
```

### Gunicorn 
+  Gunicorn is a web server.
```bash
$ sudo apt install gunicorn
$ pip install gunicorn
$ pip install gevent
```
# After installed, start your server by running:
```bash
$ gunicorn manage:app -c gc_config.py
```
+ File gc_config is used to set parameters for Gunicorn. Format of gc_config is listed as following:
```bash
bind = ":5001"
workers = 4
worker_class = "gevent"
errorlog = '-'
loglevel = "info"
```
