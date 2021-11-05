# Dockerfile

# Pulls the nvidia cuda image from docker
FROM nvidia/cuda:10.2-base

# Allows docker to cache installed dependencies between builds
COPY requirements.txt requirements.txt

# Installs python3 and the dependencies
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get -y install unzip apt-utils
RUN apt-get install python3 -y
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN python3 --version
RUN apt-get -y install python3-pip
RUN apt-get -y install ffmpeg libsm6 libxext6
RUN pip3 install --upgrade pip
RUN python3 -m pip install -U setuptools
RUN pip3 install --no-cache-dir -r requirements.txt

# Install vim
RUN ["apt-get", "install", "-y", "vim"]

# Create directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY . /usr/src/app

# Run python detect_new.py
CMD ["python3", "detect_new.py"]
