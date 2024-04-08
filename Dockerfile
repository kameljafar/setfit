FROM python:3.9-slim

RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install -y git build-essential 

COPY requirements.txt .

RUN apt-get install build-essential -y
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

COPY . .
RUN echo "firassssss"


ENV FLASK_APP=deploy.py
ARG ID_PORT
ENV FLASK_RUN_PORT=${ID_PORT}
ARG FLASK_HOST
ENV FLASK_RUN_HOST=${ID_HOST}

ARG GUNICORN_BIND
ENV GUNICORN_BIND=${GUNICORN_BIND}
ARG GUNICORN_THREADS
ENV GUNICORN_THREADS=${GUNICORN_THREADS}
ARG GUNICORN_PROCESSES
ENV GUNICORN_PROCESSES=${GUNICORN_PROCESSES}
ARG GUNICORN_TIMEOUT
ENV GUNICORN_TIMEOUT=${GUNICORN_TIMEOUT}
COPY entrypoint.sh ./entrypoint.sh
COPY gunicorn_config.py ./gunicorn_config.py
RUN chmod +x ./entrypoint.sh

RUN echo "gu"
EXPOSE 8099

CMD ["./entrypoint.sh"]
