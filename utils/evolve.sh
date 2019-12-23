#!/bin/bash
#for i in 1 2 3 4 5 6 7
#do
#  t=ultralytics/yolov3:v139 && sudo docker pull $t && sudo nvidia-docker run -d --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t utils/evolve.sh $i
#  sleep 30
# done
#
# t=ultralytics/yolov3:v139 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 320 --epochs 1 --batch-size 64 --accumulate 1 --evolve --weights '' --pre --bucket yolov4/320_coco2014_27e --device 1

while true; do
python3 train.py --data coco2014.data --img-size 320 --epochs 27 --batch-size 64 --accumulate 1 --evolve --weights '' --pre --bucket yolov4/320_coco2014_27e --device $1
done
