import os

workers = int(os.getenv('GUNICORN_PROCESSES', '2'))

threads = int(os.getenv('GUNICORN_THREADS', '2'))

timeout = int(os.getenv('GUNICORN_TIMEOUT', '8000'))

bind = os.getenv('GUNICORN_BIND', '0.0.0.0:4301')
