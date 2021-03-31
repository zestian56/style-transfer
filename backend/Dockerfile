#FROM pytorch/pytorch
FROM nvcr.io/nvidia/pytorch:21.03-py3

WORKDIR /app
ADD . /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python" , "/app/main.py"]
