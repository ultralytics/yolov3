#!/usr/bin/env bash

# New VM
rm -rf sample_data yolov3
git clone https://github.com/ultralytics/yolov3
# git clone -b test --depth 1 https://github.com/ultralytics/yolov3 test  # branch
# sudo apt-get install zip
#git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . --user && cd .. && rm -rf apex
sudo conda install -yc conda-forge scikit-image pycocotools
# python3 -c "from yolov3.utils.google_utils import gdrive_download; gdrive_download('193Zp_ye-3qXMonR1nZj3YyxMtQkMy50k','coco2014.zip')"
python3 -c "from yolov3.utils.google_utils import gdrive_download; gdrive_download('1WQT6SOktSe8Uw6r10-2JhbEhMY5DJaph','coco2017.zip')"
python3 -c "from yolov3.utils.google_utils import gdrive_download; gdrive_download('1C3HewOG9akA3y456SZLBJZfNDPkBwAto','knife.zip')"
python3 -c "from yolov3.utils.google_utils import gdrive_download; gdrive_download('13g3LqdpkNE8sPosVJT6KFXlfoMypzRP4','sm4.zip')"
sudo shutdown

# Mount local SSD
lsblk
sudo mkfs.ext4 -F /dev/nvme0n1
sudo mkdir -p /mnt/disks/nvme0n1
sudo mount /dev/nvme0n1 /mnt/disks/nvme0n1
sudo chmod a+w /mnt/disks/nvme0n1
cp -r coco /mnt/disks/nvme0n1

# Kill All
t=ultralytics/yolov3:v1
docker kill $(docker ps -a -q --filter ancestor=$t)

# Evolve coco
sudo -s
t=ultralytics/yolov3:evolve
# docker kill $(docker ps -a -q --filter ancestor=$t)
for i in 0 1 6 7; do
  docker pull $t && docker run --gpus all -d --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t bash utils/evolve.sh $i
  sleep 30
done

#COCO training
n=131 && t=ultralytics/coco:v131 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 320 640 --epochs 300 --batch 16 --weights '' --device 0 --cfg yolov3-spp.cfg --bucket ult/coco --name $n && sudo shutdown
n=132 && t=ultralytics/coco:v131 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 320 640 --epochs 300 --batch 64 --weights '' --device 0 --cfg yolov3-tiny.cfg --bucket ult/coco --name $n && sudo shutdown
