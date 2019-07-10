#!/bin/bash
# https://stackoverflow.com/questions/48133080/how-to-download-a-google-drive-url-via-curl-or-wget/48133859

# Set fileid and filename
filename="coco.zip"
fileid="1HaXkef9z6y5l4vUnCYgdmEAj61c6bfWO"  # coco2014.zip
# filename="coco.tar.gz"
# fileid="1HaXkef9z6y5l4vUnCYgdmEAj61c6bfWO"  # coco.tar.gz

# Download from Google Drive, accepting presented query
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm ./cookie

# Unzip
unzip -q ${filename}
# tar -xzf ${filename}
