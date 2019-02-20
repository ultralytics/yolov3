#!/bin/bash
# CREDIT: https://github.com/pjreddie/darknet/tree/master/scripts/get_coco_dataset.sh

# Clone COCO API
git clone https://github.com/pdollar/coco && cd coco

# Download Images
mkdir images && cd images
wget -c https://pjreddie.com/media/files/train2014.zip
wget -c https://pjreddie.com/media/files/val2014.zip

# Unzip
unzip -q train2014.zip
unzip -q val2014.zip

# (optional) Delete zip files
rm -rf *.zip

cd ..

# Download COCO Metadata
wget -c https://pjreddie.com/media/files/instances_train-val2014.zip
wget -c https://pjreddie.com/media/files/coco/5k.part
wget -c https://pjreddie.com/media/files/coco/trainvalno5k.part
wget -c https://pjreddie.com/media/files/coco/labels.tgz
tar xzf labels.tgz
unzip -q instances_train-val2014.zip

# Set Up Image Lists
paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt

# get xview training data
# wget -O train_images.tgz 'https://d307kc0mrhucc3.cloudfront.net/train_images.tgz?Expires=1530124049&Signature=JrQoxipmsETvb7eQHCfDFUO-QEHJGAayUv0i-ParmS-1hn7hl9D~bzGuHWG82imEbZSLUARTtm0wOJ7EmYMGmG5PtLKz9H5qi6DjoSUuFc13NQ-~6yUhE~NfPaTnehUdUMCa3On2wl1h1ZtRG~0Jq1P-AJbpe~oQxbyBrs1KccaMa7FK4F4oMM6sMnNgoXx8-3O77kYw~uOpTMFmTaQdHln6EztW0Lx17i57kK3ogbSUpXgaUTqjHCRA1dWIl7PY1ngQnLslkLhZqmKcaL-BvWf0ZGjHxCDQBpnUjIlvMu5NasegkwD9Jjc0ClgTxsttSkmbapVqaVC8peR0pO619Q__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ'
# tar -xvzf train_images.tgz
# sudo rm -rf train_images/._*
# lastly convert each .tif to a .bmp for faster loading in cv2

# /home/glenn_jocher3/coco/images/train2014/COCO_train2014_000000167126.jpg  # bad image??
