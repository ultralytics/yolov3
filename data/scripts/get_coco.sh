#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Download COCO 2017 dataset http://cocodataset.org
# Example usage: bash data/scripts/get_coco.sh
# parent
# ├── yolov3
# └── datasets
#     └── coco  ← downloads here

# Arguments (optional) Usage: bash data/scripts/get_coco.sh --train --val --test
if [ "$#" -gt 0 ]; then
  for opt in "$@"; do
    case "${opt}" in
      --train) train=true ;;
      --val) val=true ;;
      --test) test=true ;;
    esac
  done
else
  train=true
  val=true
  test=false
fi

# Download/unzip labels
d='../datasets' # unzip directory
url=https://github.com/ultralytics/assets/releases/download/v0.0.0/
f='coco2017labels.zip' # 46 MB
echo 'Downloading' $url$f ' ...'
curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &

# Download/unzip images
d='../datasets/coco/images' # unzip directory
url=http://images.cocodataset.org/zips/
if [ "$train" == "true" ]; then
  f='train2017.zip' # 19G, 118k images
  echo 'Downloading' $url$f '...'
  curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &
fi
if [ "$val" == "true" ]; then
  f='val2017.zip' # 1G, 5k images
  echo 'Downloading' $url$f '...'
  curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &
fi
if [ "$test" == "true" ]; then
  f='test2017.zip' # 7G, 41k images (optional)
  echo 'Downloading' $url$f '...'
  curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &
fi
wait # finish background tasks
