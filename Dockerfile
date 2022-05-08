FROM python:3.8

RUN pwd
WORKDIR /projeto_maroto
RUN pwd

COPY requirements_v1.txt .
RUN pip install -r requirements_v1.txt

RUN ls -la
COPY . .
RUN ls -la

WORKDIR src

CMD ["python", "train.py"]