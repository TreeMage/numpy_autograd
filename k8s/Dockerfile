FROM python:3.8-slim-buster

COPY . /app
COPY k8s/entrypoint.sh /app
WORKDIR /app

RUN pip install -r requirements.txt

RUN chmod ugo+rx /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
