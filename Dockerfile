# Start from Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:19.07-py3

# Install dependencies
RUN pip3 install -U -r requirements.txt

# Move file into container
# RUN mv 1047.tif ./1047.tif

# Move model into container
# RUN mv yolov3-spp.pt ./weights


# ---------------------------------------------------  Extras Below  ---------------------------------------------------

# Build container
# sudo docker image prune -a && sudo docker build -t friendlyhello . && sudo docker tag friendlyhello ultralytics/yolov3:v0

# Run container
# time sudo docker run -it --memory=8g --cpus=4 ultralytics/yolov3:v0 bash -c './run.sh /1047.tif /tmp && cat /tmp/1047.tif.txt'

# Push container to https://hub.docker.com/u/ultralytics
# sudo docker push ultralytics/xview:v30