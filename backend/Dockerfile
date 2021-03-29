#FROM pytorch/pytorch
FROM nvcr.io/nvidia/pytorch:20.06-py3

WORKDIR /app
ADD . /app

RUN ln -s /usr/bin/python3 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python" , "/app/main.py"]