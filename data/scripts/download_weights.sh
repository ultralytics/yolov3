#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Download latest models from https://github.com/ultralytics/yolov3/releases
# Example usage: bash data/scripts/download_weights.sh
# parent
# └── yolov3
#     ├── yolov3.pt  ← downloads here
#     ├── yolov3-spp.pt
#     └── ...

python - << EOF
from utils.downloads import attempt_download

for x in 'yolov3', 'yolov3-spp', 'yolov3-tiny':
    attempt_download(f'weights/{x}.pt')

EOF
