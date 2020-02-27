FROM tensorflow/tensorflow:latest-gpu-py3
COPY . /app

RUN pip3 install --upgrade pip
RUN pip3 install -r /app/requirements.txt

CMD ["bash", "-c", "source /etc/bash.bashrc"]