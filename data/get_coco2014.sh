#!/bin/bash
# Zip coco folder
# zip -r coco.zip coco
# tar -czvf coco.tar.gz coco

# Download labels from Google Drive, accepting presented query
filename="coco2014labels.zip"
fileid="1s6-CmF5_SElM28r52P1OUrCcuXZN-SFo"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" >/dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=$(awk '/download/ {print $NF}' ./cookie)&id=${fileid}" -o ${filename}
rm ./cookie

# Unzip labels
unzip -q ${filename} # for coco.zip
# tar -xzf ${filename} # for coco.tar.gz
rm ${filename}

# Download and unzip images
cd coco/images
f="train2014.zip" && curl http://images.cocodataset.org/zips/$f -o $f && unzip -q $f && rm $f
f="val2014.zip" && curl http://images.cocodataset.org/zips/$f -o $f && unzip -q $f && rm $f

# cd out
cd ../..
