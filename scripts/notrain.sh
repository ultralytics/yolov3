python train.py --data coco.yaml \
                --cfg models/yolov3-quan-4-6-10bit_final.yaml \
                --weights /home/zhangyifan/yolov3/runs/train/exp76/weights/best.pt \
                --hyp /home/zhangyifan/yolov3/data/hyps/hyp.scratch_notrain.yaml \
                --batch-size 16\
                --device 2 \
                --img 416 \
