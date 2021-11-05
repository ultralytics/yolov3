#!/bin/sh -v

echo 'building yolov3 image'
docker build -f NoGPU.Dockerfile -t yolov3nogpu .
wait

echo 'running yolov3container container'
docker run -dit --name yolov3nogpucontainer yolov3nogpu
docker logs --follow yolov3nogpucontainer
wait

echo 'copying from docker to local'
docker start yolov3nogpucontainer
docker cp yolov3nogpucontainer:/usr/src/app/Assignment3/Results/. ./Assignment3/Results
wait

echo 'killing container'
docker kill yolov3nogpucontainer
wait

echo 'remove container'
docker rm yolov3nogpucontainer