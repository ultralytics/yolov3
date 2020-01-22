#!/bin/bash
#for i in 0 1 2 3
#do
#  t=ultralytics/yolov3:v139 && sudo docker pull $t && sudo nvidia-docker run -d --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t utils/evolve.sh $i
#  sleep 30
#done

while true; do
  # python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --multi --bucket ult/wer --evolve --cache --device $1 --cfg yolov3-tiny-1cls.cfg --single --adam
  # python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --multi --bucket ult/athena --evolve --device $1 --cfg yolov3-spp-1cls.cfg

  # python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --evolve --weights '' --bucket yolov4/640ms_coco2014_10e --device $1 --multi
  # python3 train.py --data coco2014.data --img-size 320 --epochs 27 --batch 64 --accum 1 --evolve --weights '' --bucket yolov4/320_coco2014_27e --device $1
  python3 train.py --data coco2014.data --img-size 384 --epochs 27 --batch 64 --accum 1 --evolve --weights '' --bucket ult/coco --device $1 --cache
done
