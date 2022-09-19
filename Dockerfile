FROM python:3.8.5

ADD ./ /home
RUN python -m pip install --upgrade pip
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -e home/

CMD jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
