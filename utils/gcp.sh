#!/usr/bin/env bash

# New VM
rm -rf sample_data yolov3 darknet apex coco cocoapi knife knifec
git clone https://github.com/ultralytics/yolov3
git clone https://github.com/AlexeyAB/darknet && cd darknet && make GPU=1 CUDNN=1 CUDNN_HALF=1 OPENCV=1 && wget -c https://pjreddie.com/media/files/darknet53.conv.74 && cd ..
git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . --user && cd .. && rm -rf apex
python3 -c "
from yolov3.utils.google_utils import gdrive_download
gdrive_download('1HaXkef9z6y5l4vUnCYgdmEAj61c6bfWO','coco.zip')
gdrive_download('1GrFcTIIsKzOafZltUOS75RSahPrj2KyT','knife.zip')
gdrive_download('19sLJEGHlIAIFHcEftq4aLCw_tkWZmhD1','knifec.zip')"
sudo shutdown

# Re-clone
rm -rf yolov3  # Warning: remove existing
git clone https://github.com/ultralytics/yolov3  # master
# git clone -b test --depth 1 https://github.com/ultralytics/yolov3 test  # branch
cp -r cocoapi/PythonAPI/pycocotools yolov3
cp -r weights yolov3 && cd yolov3

# Train
python3 train.py

# Resume
python3 train.py --resume

# Detect
python3 detect.py

# Test
python3 test.py --save-json

# Evolve
for i in {0..500}
do
  python3 train.py --data data/coco.data --img-size 320 --epochs 1 --batch-size 64 --accumulate 1 --evolve --bucket yolov4
done

# Git pull
git pull https://github.com/ultralytics/yolov3  # master
git pull https://github.com/ultralytics/yolov3 test  # branch

# Test Darknet training
python3 test.py --weights ../darknet/backup/yolov3.backup

# Copy last.pt TO bucket
gsutil cp yolov3/weights/last1gpu.pt gs://ultralytics

# Copy last.pt FROM bucket
gsutil cp gs://ultralytics/last.pt yolov3/weights/last.pt
wget https://storage.googleapis.com/ultralytics/yolov3/last_v1_0.pt -O weights/last_v1_0.pt
wget https://storage.googleapis.com/ultralytics/yolov3/best_v1_0.pt -O weights/best_v1_0.pt

# Reproduce tutorials
rm results*.txt  # WARNING: removes existing results
python3 train.py --nosave --data data/coco_1img.data && mv results.txt results0r_1img.txt
python3 train.py --nosave --data data/coco_10img.data && mv results.txt results0r_10img.txt
python3 train.py --nosave --data data/coco_100img.data && mv results.txt results0r_100img.txt
# python3 train.py --nosave --data data/coco_100img.data --transfer && mv results.txt results3_100imgTL.txt
python3 -c "from utils import utils; utils.plot_results()"
# gsutil cp results*.txt gs://ultralytics
gsutil cp results.png gs://ultralytics
sudo shutdown

# Reproduce mAP
python3 test.py --save-json --img-size 608
python3 test.py --save-json --img-size 416
python3 test.py --save-json --img-size 320
sudo shutdown

# Benchmark script
git clone https://github.com/ultralytics/yolov3  # clone our repo
git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . --user && cd .. && rm -rf apex  # install nvidia apex
python3 -c "from yolov3.utils.google_utils import gdrive_download; gdrive_download('1HaXkef9z6y5l4vUnCYgdmEAj61c6bfWO','coco.zip')"  # download coco dataset (20GB)
cd yolov3 && clear && python3 train.py --epochs 1  # run benchmark (~30 min)

# Unit tests
python3 detect.py  # detect 2 persons, 1 tie
python3 test.py --data data/coco_32img.data  # test mAP = 0.8
python3 train.py --data data/coco_32img.data --epochs 5 --nosave  # train 5 epochs
python3 train.py --data data/coco_1cls.data --epochs 5 --nosave  # train 5 epochs
python3 train.py --data data/coco_1img.data --epochs 5 --nosave  # train 5 epochs

# AlexyAB Darknet
gsutil cp -r gs://sm6/supermarket2 .  # dataset from bucket
rm -rf darknet && git clone https://github.com/AlexeyAB/darknet && cd darknet && wget -c https://pjreddie.com/media/files/darknet53.conv.74  # sudo apt install libopencv-dev && make
./darknet detector calc_anchors data/coco_img64.data -num_of_clusters 9 -width 320 -height 320  # kmeans anchor calculation
./darknet detector train ../supermarket2/supermarket2.data ../yolo_v3_spp_pan_scale.cfg darknet53.conv.74 -map -dont_show # train spp
./darknet detector train ../yolov3/data/coco.data ../yolov3-spp.cfg darknet53.conv.74 -map -dont_show # train spp coco

./darknet detector train data/coco.data ../yolov3-spp.cfg darknet53.conv.74 -map -dont_show # train spp
gsutil cp -r backup/*5000.weights gs://sm6/weights
sudo shutdown


./darknet detector train ../supermarket2/supermarket2.data ../yolov3-tiny-sm2-1cls.cfg yolov3-tiny.conv.15 -map -dont_show # train tiny
./darknet detector train ../supermarket2/supermarket2.data cfg/yolov3-spp-sm2-1cls.cfg backup/yolov3-spp-sm2-1cls_last.weights  # resume
python3 train.py --data ../supermarket2/supermarket2.data --cfg ../yolov3-spp-sm2-1cls.cfg --epochs 100 --num-workers 8 --img-size 320 --nosave  # train ultralytics
python3 test.py --data ../supermarket2/supermarket2.data --weights ../darknet/backup/yolov3-spp-sm2-1cls_5000.weights --cfg cfg/yolov3-spp-sm2-1cls.cfg  # test
gsutil cp -r backup/*.weights gs://sm6/weights  # weights to bucket

python3 test.py --data ../supermarket2/supermarket2.data --weights weights/yolov3-spp-sm2-1cls_5000.weights --cfg ../yolov3-spp-sm2-1cls.cfg --img-size 320 --conf-thres 0.2  # test
python3 test.py --data ../supermarket2/supermarket2.data --weights weights/yolov3-spp-sm2-1cls-scalexy_125_5000.weights --cfg ../yolov3-spp-sm2-1cls-scalexy_125.cfg --img-size 320 --conf-thres 0.2  # test
python3 test.py --data ../supermarket2/supermarket2.data --weights weights/yolov3-spp-sm2-1cls-scalexy_150_5000.weights --cfg ../yolov3-spp-sm2-1cls-scalexy_150.cfg --img-size 320 --conf-thres 0.2  # test
python3 test.py --data ../supermarket2/supermarket2.data --weights weights/yolov3-spp-sm2-1cls-scalexy_200_5000.weights --cfg ../yolov3-spp-sm2-1cls-scalexy_200.cfg --img-size 320 --conf-thres 0.2  # test
python3 test.py --data ../supermarket2/supermarket2.data --weights ../darknet/backup/yolov3-spp-sm2-1cls-scalexy_variable_5000.weights --cfg ../yolov3-spp-sm2-1cls-scalexy_variable.cfg --img-size 320 --conf-thres 0.2  # test


# Debug/Development
python3 train.py --data data/coco.data --img-size 320 --single-scale --batch-size 64 --accumulate 1 --epochs 1 --evolve --giou
python3 test.py --weights weights/last.pt --cfg cfg/yolov3-spp.cfg --img-size 320

gsutil cp evolve.txt gs://ultralytics
sudo shutdown
