#!/usr/bin/env bash

# New VM
rm -rf sample_data yolov3 darknet apex coco cocoapi knife knifec
git clone https://github.com/ultralytics/yolov3
# git clone https://github.com/AlexeyAB/darknet && cd darknet && make GPU=1 CUDNN=1 CUDNN_HALF=1 OPENCV=0 && wget -c https://pjreddie.com/media/files/darknet53.conv.74 && cd ..
git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . --user && cd .. && rm -rf apex
sudo conda install -y -c conda-forge scikit-image pycocotools # tensorboard
python3 -c "
from yolov3.utils.google_utils import gdrive_download
gdrive_download('1HaXkef9z6y5l4vUnCYgdmEAj61c6bfWO','coco.zip')"
sudo shutdown

# Re-clone
rm -rf yolov3  # Warning: remove existing
git clone https://github.com/ultralytics/yolov3 && cd yolov3 # master
# git clone -b test --depth 1 https://github.com/ultralytics/yolov3 test  # branch
python3 train.py --img-size 320 --weights weights/darknet53.conv.74 --epochs 27 --batch-size 64 --accumulate 1

# Train
python3 train.py

# Resume
python3 train.py --resume

# Detect
python3 detect.py

# Test
python3 test.py --save-json

# Evolve
export t=ultralytics/yolov3:v0 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t
clear
sleep 200
while true
do
  python3 train.py --data data/coco.data --img-size 416 --epochs 27 --batch-size 32 --accumulate 2 --evolve --weights '' --prebias --bucket yolov4/416_coco_27e --device 7
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

#Docker
sudo docker kill "$(sudo docker ps -q)"
sudo docker pull ultralytics/yolov3:v0
sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco ultralytics/yolov3:v0


export t=ultralytics/yolov3:v70 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 70 --device 0 --multi
export t=ultralytics/yolov3:v0 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 71 --device 0 --multi --img-weights

export t=ultralytics/yolov3:v73 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 73 --device 5 --cfg cfg/yolov3s.cfg
export t=ultralytics/yolov3:v74 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 74 --device 0 --cfg cfg/yolov3s.cfg
export t=ultralytics/yolov3:v75 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 75 --device 7 --cfg cfg/yolov3s.cfg
export t=ultralytics/yolov3:v76 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 76 --device 0 --cfg cfg/yolov3-spp.cfg

export t=ultralytics/yolov3:v79 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 79 --device 5
export t=ultralytics/yolov3:v80 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 80 --device 0
export t=ultralytics/yolov3:v81 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 81 --device 7
export t=ultralytics/yolov3:v82 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 82 --device 0 --cfg cfg/yolov3s.cfg

export t=ultralytics/yolov3:v83 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 83 --device 1 --multi
export t=ultralytics/yolov3:v84 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 84 --device 0 --multi
export t=ultralytics/yolov3:v85 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 85 --device 0 --multi
export t=ultralytics/yolov3:v86 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 86 --device 1 --multi
export t=ultralytics/yolov3:v87 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 87 --device 2 --multi
export t=ultralytics/yolov3:v88 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 88 --device 3 --multi
export t=ultralytics/yolov3:v89 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 89 --device 1
export t=ultralytics/yolov3:v90 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 90 --device 0 --cfg cfg/yolov3-spp-matrix.cfg
export t=ultralytics/yolov3:v91 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 91 --device 0 --cfg cfg/yolov3-spp-matrix.cfg


export t=ultralytics/yolov3:v92 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch-size 16 --accumulate 4 --prebias --bucket yolov4 --name 92 --device 0




#SM4
export t=ultralytics/yolov3:v0 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host --mount type=bind,source="$(pwd)"/data,target=/usr/src/data $t python3 train.py --weights 'ultralytics49.pt' --epochs 500 --img-size 320 --batch-size 32 --accumulate 2 --prebias --bucket yolov4 --name 78 --device 0 --multi --cfg cfg/yolov3-spp-3cls.cfg --data ../data/sm4/out.data


export t=ultralytics/yolov3:v2 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t
clear
sleep 120
while true
do
  python3 train.py --weights '' --epochs 27 --batch-size 32 --accumulate 2 --prebias --evolve --device 7 --bucket yolov4/416_coco_27e
done


while true; do python3 train.py --data data/coco.data --img-size 320 --batch-size 64 --accumulate 1 --evolve --epochs 1 --adam --bucket yolov4/adamdefaultpw_coco_1e; done






