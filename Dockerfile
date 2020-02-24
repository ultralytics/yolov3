# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:20.01-py3

# Install dependencies (pip or conda)
RUN pip install -U gsutil
# RUN pip install -U -r requirements.txt
# RUN conda update -n base -c defaults conda
# RUN conda install -y -c anaconda future numpy opencv matplotlib tqdm pillow
# RUN conda install -y -c conda-forge scikit-image tensorboard pycocotools

## Install OpenCV with Gstreamer support
#WORKDIR /usr/src
#RUN pip uninstall -y opencv-python
#RUN apt-get update
#RUN apt-get install -y gstreamer1.0-tools gstreamer1.0-python3-dbg-plugin-loader libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
#RUN git clone https://github.com/opencv/opencv.git && cd opencv && git checkout 4.1.1 && mkdir build
#RUN git clone https://github.com/opencv/opencv_contrib.git && cd opencv_contrib && git checkout 4.1.1
#RUN cd opencv/build && cmake ../ \
#    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
#    -D BUILD_OPENCV_PYTHON3=ON \
#    -D PYTHON3_EXECUTABLE=/opt/conda/bin/python \
#    -D PYTHON3_INCLUDE_PATH=/opt/conda/include/python3.6m \
#    -D PYTHON3_LIBRARIES=/opt/conda/lib/python3.6/site-packages \
#    -D WITH_GSTREAMER=ON \
#    -D WITH_FFMPEG=OFF \
#    && make && make install && ldconfig
#RUN cd /usr/local/lib/python3.6/site-packages/cv2/python-3.6/ && mv cv2.cpython-36m-x86_64-linux-gnu.so cv2.so
#RUN cd /opt/conda/lib/python3.6/site-packages/ && ln -s /usr/local/lib/python3.6/site-packages/cv2/python-3.6/cv2.so cv2.so
#RUN python3 -c "import cv2; print(cv2.getBuildInformation())"

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# Copy weights
#RUN python3 -c "from models import *; \
#attempt_download('weights/yolov3.pt'); \
#attempt_download('weights/yolov3-spp.pt')"


# ---------------------------------------------------  Extras Below  ---------------------------------------------------

# Build and Push
# t=ultralytics/yolov3:v0 && sudo docker build -t $t . && sudo docker push $t

# Run
# t=ultralytics/yolov3:v0 && sudo docker pull $t && sudo docker run -it $t bash

# Pull and Run with local directory access
# t=ultralytics/yolov3:v0 && sudo docker pull $t && sudo docker run -it -v "$(pwd)"/coco:/usr/src/coco $t bash

# Kill all
# sudo docker kill "$(sudo docker ps -q)"

# Kill all image-based
# sudo docker kill $(sudo docker ps -a -q --filter ancestor=ultralytics/yolov3:v0)

# Run bash for loop
# sudo docker run --gpus all --ipc=host ultralytics/yolov3:v0 while true; do python3 train.py --evolve; done
