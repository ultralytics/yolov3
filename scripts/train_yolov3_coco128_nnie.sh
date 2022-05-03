python train.py --data coco128.yaml \
                --weights yolov3.pt \
                --batch-size 8 \
                --device 1 \
                --img 640 \
                --quantize \
                --BackendType NNIE \
