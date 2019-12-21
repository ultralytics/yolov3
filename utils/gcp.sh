#!/usr/bin/env bash

# New VM
rm -rf sample_data yolov3
git clone https://github.com/ultralytics/yolov3
git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . --user && cd .. && rm -rf apex
sudo conda install -yc conda-forge scikit-image pycocotools
python3 -c "from yolov3.utils.google_utils import gdrive_download; gdrive_download('193Zp_ye-3qXMonR1nZj3YyxMtQkMy50k','coco2014.zip')"
python3 -c "from yolov3.utils.google_utils import gdrive_download; gdrive_download('1WQT6SOktSe8Uw6r10-2JhbEhMY5DJaph','coco2017.zip')"
sudo shutdown

# Re-clone
rm -rf yolov3  # Warning: remove existing
git clone https://github.com/ultralytics/yolov3 # master
bash yolov3/data/get_coco2017.sh
# git clone -b test --depth 1 https://github.com/ultralytics/yolov3 test  # branch
cd yolov3
python3 train.py --weights '' --epochs 27 --batch-size 32 --accumulate 2 --nosave --data coco2017.data

# Train
python3 train.py

# Resume
python3 train.py --resume

# Detect
python3 detect.py

# Test
python3 test.py --save-json

# Evolve
for i in 1 2 3 4 5 6 7
do
  export t=ultralytics/yolov3:v139 && sudo docker pull $t && sudo nvidia-docker run -d --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 320 --epochs 27 --batch-size 64 --accumulate 1 --evolve --weights '' --pre --arc default --bucket yolov4/320_coco2014_27e --device $i
  sleep 30
done

export t=ultralytics/yolov3:v139 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco16.data --img-size 320 --epochs 1 --batch-size 8 --accumulate 1 --evolve --weights '' --device 1


export t=ultralytics/yolov3:v139 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t
clear
sleep 0
while true
do
  python3 train.py --data coco2014.data --img-size 416 --epochs 27 --batch-size 32 --accumulate 2 --evolve --weights '' --pre --bucket yolov4/416_coco_27e --device 2
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


export t=ultralytics/yolov3:v70 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 70 --device 0 --multi
export t=ultralytics/yolov3:v0 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 71 --device 0 --multi --img-weights

export t=ultralytics/yolov3:v73 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 73 --device 5 --cfg cfg/yolov3s.cfg
export t=ultralytics/yolov3:v74 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 74 --device 0 --cfg cfg/yolov3s.cfg
export t=ultralytics/yolov3:v75 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 75 --device 7 --cfg cfg/yolov3s.cfg
export t=ultralytics/yolov3:v76 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 76 --device 0 --cfg cfg/yolov3-spp.cfg

export t=ultralytics/yolov3:v79 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 79 --device 5
export t=ultralytics/yolov3:v80 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 80 --device 0
export t=ultralytics/yolov3:v81 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 81 --device 7
export t=ultralytics/yolov3:v82 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 82 --device 0 --cfg cfg/yolov3s.cfg

export t=ultralytics/yolov3:v83 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 83 --device 6 --multi --nosave
export t=ultralytics/yolov3:v84 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 84 --device 0 --multi
export t=ultralytics/yolov3:v85 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 85 --device 0 --multi
export t=ultralytics/yolov3:v86 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 86 --device 1 --multi
export t=ultralytics/yolov3:v87 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 87 --device 2 --multi
export t=ultralytics/yolov3:v88 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 88 --device 3 --multi
export t=ultralytics/yolov3:v89 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 89 --device 1
export t=ultralytics/yolov3:v90 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 90 --device 0 --cfg cfg/yolov3-spp-matrix.cfg
export t=ultralytics/yolov3:v91 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 91 --device 0 --cfg cfg/yolov3-spp-matrix.cfg

export t=ultralytics/yolov3:v92 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 92 --device 0
export t=ultralytics/yolov3:v93 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 93 --device 0 --cfg cfg/yolov3-spp-matrix.cfg


#SM4
export t=ultralytics/yolov3:v96 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'ultralytics68.pt' --epochs 1000 --img-size 320 --batch 32 --accum 2 --pre --bucket yolov4 --name 96 --device 0 --multi --cfg cfg/yolov3-spp-3cls.cfg --data ../data/sm4/out.data --nosave
export t=ultralytics/yolov3:v97 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'ultralytics68.pt' --epochs 1000 --img-size 320 --batch 32 --accum 2 --pre --bucket yolov4 --name 97 --device 4 --multi --cfg cfg/yolov3-spp-3cls.cfg --data ../data/sm4/out.data --nosave
export t=ultralytics/yolov3:v98 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'ultralytics68.pt' --epochs 1000 --img-size 320 --batch 16 --accum 4 --pre --bucket yolov4 --name 98 --device 5 --multi --cfg cfg/yolov3-spp-3cls.cfg --data ../data/sm4/out.data --nosave
export t=ultralytics/yolov3:v113 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 16 --accum 4 --pre --bucket yolov4 --name 101 --device 7 --multi --nosave

export t=ultralytics/yolov3:v102 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'yolov3-tiny.pt' --epochs 1000 --img-size 320 --batch 64 --accum 1 --pre --bucket yolov4 --name 102 --device 0 --cfg cfg/yolov3-tiny-3cls.cfg --data ../data/sm4/out.data --nosave --cache
export t=ultralytics/yolov3:v103 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'yolov3-tiny.pt' --epochs 500 --img-size 320 --batch 64 --accum 1 --pre --bucket yolov4 --name 103 --device 0 --cfg cfg/yolov3-tiny-3cls.cfg --data ../data/sm4/out.data --nosave --cache
export t=ultralytics/yolov3:v104 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'yolov3-tiny.pt' --epochs 500 --img-size 320 --batch 64 --accum 1 --pre --bucket yolov4 --name 104 --device 0 --cfg cfg/yolov3-tiny-3cls.cfg --data ../data/sm4/out.data --nosave --cache
export t=ultralytics/yolov3:v105 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'yolov3-tiny.pt' --epochs 500 --img-size 320 --batch 64 --accum 1 --pre --bucket yolov4 --name 105 --device 0 --cfg cfg/yolov3-tiny-3cls.cfg --data ../data/sm4/out.data --nosave --cache
export t=ultralytics/yolov3:v106 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'yolov3-tiny.pt' --epochs 500 --img-size 320 --batch 64 --accum 1 --pre --bucket yolov4 --name 106 --device 0 --cfg cfg/yolov3-tiny-3cls-sm4.cfg --data ../data/sm4/out.data --nosave --cache
export t=ultralytics/yolov3:v107 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 107 --device 5 --nosave --cfg cfg/yolov3-spp3.cfg
export t=ultralytics/yolov3:v108 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 108 --device 7 --nosave

export t=ultralytics/yolov3:v109 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 109 --device 4 --multi --nosave
export t=ultralytics/yolov3:v110 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accumulate 4 --pre --bucket yolov4 --name 110 --device 3 --multi --nosave

export t=ultralytics/yolov3:v83 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 111 --device 0
export t=ultralytics/yolov3:v112 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 112 --device 1 --nosave
export t=ultralytics/yolov3:v113 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 113 --device 2 --nosave
export t=ultralytics/yolov3:v114 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 114 --device 2 --nosave
export t=ultralytics/yolov3:v113 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 115 --device 5 --nosave  --cfg cfg/yolov3-spp3.cfg
export t=ultralytics/yolov3:v116 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 116 --device 1 --nosave

export t=ultralytics/yolov3:v83 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 16 --accum 4 --epochs 27 --pre --bucket yolov4 --name 117 --device 0 --nosave --multi
export t=ultralytics/yolov3:v118 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 16 --accum 4 --epochs 27 --pre --bucket yolov4 --name 118 --device 5 --nosave --multi
export t=ultralytics/yolov3:v119 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 119 --device 1 --nosave
export t=ultralytics/yolov3:v120 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 120 --device 2 --nosave
export t=ultralytics/yolov3:v121 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 121 --device 0 --nosave --cfg cfg/csresnext50-panet-spp.cfg
export t=ultralytics/yolov3:v122 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 273 --pre --bucket yolov4 --name 122 --device 2 --nosave
export t=ultralytics/yolov3:v123 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 273 --pre --bucket yolov4 --name 123 --device 5 --nosave

export t=ultralytics/yolov3:v124 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 124 --device 0 --nosave --cfg yolov3-tiny
export t=ultralytics/yolov3:v124 && sudo docker pull $t && sudo nvidia-docker run -d --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 125 --device 1 --nosave --cfg yolov3-tiny2
export t=ultralytics/yolov3:v124 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 126 --device 1 --nosave --cfg yolov3-tiny3
export t=ultralytics/yolov3:v127 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 127 --device 0 --nosave --cfg yolov3-tiny4
export t=ultralytics/yolov3:v124 && sudo docker pull $t && sudo nvidia-docker run -d --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 64 --accum 1 --epochs 273 --pre --bucket yolov4 --name 128 --device 1 --nosave --cfg yolov3-tiny2 --multi
export t=ultralytics/yolov3:v129 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 64 --accum 1 --epochs 273 --pre --bucket yolov4 --name 129 --device 0 --nosave --cfg yolov3-tiny2

export t=ultralytics/yolov3:v130 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 130 --device 0 --nosave
export t=ultralytics/yolov3:v133 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 22 --accum 3 --epochs 250 --pre --bucket yolov4 --name 131 --device 0 --nosave --multi
export t=ultralytics/yolov3:v130 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 132 --device 0 --nosave --data coco2014.data
export t=ultralytics/yolov3:v133 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 22 --accum 3 --epochs 27 --pre --bucket yolov4 --name 133 --device 0 --nosave --multi
export t=ultralytics/yolov3:v134 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 134 --device 0 --nosave --data coco2014.data

export t=ultralytics/yolov3:v135 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 24 --accum 3 --epochs 270 --pre --bucket yolov4 --name 135 --device 0 --nosave --multi --data coco2014.data
export t=ultralytics/yolov3:v136 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 24 --accum 3 --epochs 270 --pre --bucket yolov4 --name 136 --device 0 --nosave --multi --data coco2014.data

export t=ultralytics/yolov3:v137 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 137 --device 7 --nosave --data coco2014.data
export t=ultralytics/yolov3:v137 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --bucket yolov4 --name 138 --device 6 --nosave --data coco2014.data

export t=ultralytics/yolov3:v140 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 140 --device 1 --nosave --data coco2014.data --arc uBCE
export t=ultralytics/yolov3:v141 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 141 --device 0 --nosave --data coco2014.data --arc uBCE
export t=ultralytics/yolov3:v142 && sudo docker pull $t && sudo nvidia-docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 142 --device 1 --nosave --data coco2014.data --arc uBCE


export t=ultralytics/yolov3:v139 && sudo docker build -t $t . && sudo docker push $t

conda update -n base -c defaults conda
conda install -yc anaconda numpy opencv matplotlib tqdm pillow ipython future
conda install -yc conda-forge scikit-image pycocotools onnx tensorboard
conda install -yc spyder-ide spyder-line-profiler
conda install -yc pytorch pytorch torchvision