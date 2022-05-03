python train.py --data coco128.yaml \
                --weights yolov3.pt \
                --device cpu \
                --img 640 \
                --quantize \
                --BackendType PPLW8A16 \
