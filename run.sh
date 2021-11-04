#!/bin/sh -v

echo 'building yolov3 image'
docker build -t yolov3 .
wait

echo 'running yolov3container container'
docker run -dit --name yolov3container --gpus all yolov3
docker logs --follow yolov3container
wait

echo 'copying from docker to local'
docker start yolov3container
docker cp yolov3container:/usr/src/app/Assignment3/Results/. ./Assignment3/Results
wait

echo 'killing container'
docker kill yolov3container
wait

echo 'remove container'
docker rm yolov3container