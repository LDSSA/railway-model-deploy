FROM python:3.14

ADD . /opt/ml_in_app
WORKDIR /opt/ml_in_app

# install production packages with pip
RUN pip install -r requirements_prod.txt
CMD ["python", "app.py"]
