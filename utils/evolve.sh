#!/bin/bash
#for i in 0 1 2 3
#do
#  t=ultralytics/yolov3:v139 && sudo docker pull $t && sudo nvidia-docker run -d --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t utils/evolve.sh $i
#  sleep 30
#done

while true; do
  # python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.conv.15 --multi --bucket ult/wer --evolve --cache --device $1 --cfg yolov3-tiny3-1cls.cfg --single --adam
  # python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --multi --bucket ult/athena --evolve --device $1 --cfg yolov3-spp-1cls.cfg

  python3 train.py --data coco2014.data --img-size 512 608 --epochs 27 --batch 8 --accum 8 --evolve --weights '' --bucket ult/coco/sppa_512 --device $1 --cfg yolov3-sppa.cfg --multi
done

# coco epoch times --img-size 416 608 --epochs 27 --batch 16 --accum 4
# 36:34 2080ti
# 21:58 V100
# 63:00 T4
