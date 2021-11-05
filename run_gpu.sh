#!/bin/sh -v

echo 'building yolov3 image'
docker build -f GPU.Dockerfile -t yolov3gpu .
wait

echo 'running yolov3container container'
docker run -dit --name yolov3gpucontainer --gpus all yolov3gpu
docker logs --follow yolov3gpucontainer
wait

echo 'copying from docker to local'
docker start yolov3gpucontainer
docker cp yolov3gpucontainer:/usr/src/app/Assignment3/Results/. ./Assignment3/Results
wait

echo 'killing container'
docker kill yolov3gpucontainer
wait

echo 'remove container'
docker rm yolov3gpucontainer