import glob
import os
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

# cv2.imread() jpg at 230 img/s, *.bmp at 400 img/s
for path in ['../coco/images/val2014/', '../coco/images/train2014/']:
    folder = os.sep + Path(path).name
    output = path.replace(folder, folder + 'bmp')
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    for f in tqdm(glob.glob('%s*.jpg' % path)):
        save_name = f.replace('.jpg', '.bmp').replace(folder, folder + 'bmp')
        cv2.imwrite(save_name, cv2.imread(f))
