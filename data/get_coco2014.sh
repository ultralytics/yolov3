#!/bin/bash
# Zip coco folder
# zip -r coco.zip coco
# tar -czvf coco.tar.gz coco

# Download labels from Google Drive, accepting presented query
filename="coco2014labels.zip"
fileid="1s6-CmF5_SElM28r52P1OUrCcuXZN-SFo"

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm ./cookie

# Unzip labels
unzip -q ${filename}  # for coco.zip
# tar -xzf ${filename}  # for coco.tar.gz
rm ${filename}

# Download images
cd coco/images
wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip

# Unzip images
unzip -q train2014.zip
unzip -q val2014.zip

# (optional) Delete zip files
rm -rf *.zip

# cd out
cd ../..

