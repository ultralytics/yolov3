#!/bin/bash
# Download latest models from https://github.com/ultralytics/yolov3/releases

python - <<EOF
from utils.google_utils import attempt_download

for x in ['yolov3', 'yolov3-spp', 'yolov3-tiny']:
    attempt_download(f'{x}.pt')

EOF
