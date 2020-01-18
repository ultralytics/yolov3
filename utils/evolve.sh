#!/bin/bash
#for i in 1 2 3 4 5 6 7
#do
#  t=ultralytics/yolov3:v139 && sudo docker pull $t && sudo nvidia-docker run -d --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t utils/evolve.sh $i
#  sleep 30
# done
#
#t=ultralytics/yolov3:v199 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 672 --epochs 10 --batch 16 --accum 4 --weights '' --arc defaultpw --device 0 --multi

while true; do
  # python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc default --pre --multi --bucket ult/wer --evolve --device $1 --cfg yolov3-tiny-3cls.cfg --cache
  python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --arc default --pre --multi --bucket ult/athena --evolve --device $1 --cfg yolov3-spp-1cls.cfg

  # python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --evolve --weights '' --arc defaultpw --pre --bucket yolov4/640ms_coco2014_10e --device $1 --multi
  # python3 train.py --data coco2014.data --img-size 320 --epochs 27 --batch 64 --accum 1 --evolve --weights '' --arc defaultpw --pre --bucket yolov4/320_coco2014_27e --device $1
done
