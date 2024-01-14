FROM python:3.9.12-slim-buster
WORKDIR /code
COPY . /code
RUN pip install -r requirements.txt
#RUN -p 5000:5000 --name my-flask-app flask-app
EXPOSE 5000
CMD ["python", "objectDetectionServer.py"]
