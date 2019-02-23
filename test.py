import argparse

from models import *
from utils.datasets import *
from utils.utils import *


def test(
        cfg,
        data_cfg,
        weights,
        batch_size=16,
        img_size=416,
        iou_thres=0.5,
        conf_thres=0.3,
        nms_thres=0.45
):
    device = torch_utils.select_device()

    # Configure run
    data_cfg_dict = parse_data_cfg(data_cfg)
    nC = int(data_cfg_dict['classes'])  # number of classes (80 for COCO)
    test_path = data_cfg_dict['valid']

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    model.to(device).eval()

    # Get dataloader
    # dataloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path), batch_size=batch_size)  # pytorch
    dataloader = LoadImagesAndLabels(test_path, batch_size=batch_size, img_size=img_size)

    mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class = [], [], [], [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)
    for batch_i, (imgs, targets) in enumerate(dataloader):
        output = model(imgs.to(device))
        output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)

        # Compute average precision for each sample
        for sample_i, (labels, detections) in enumerate(zip(targets, output)):
            seen += 1

            if detections is None:
                # If there are labels but no detections mark as zero AP
                if labels.size(0) != 0:
                    mAPs.append(0), mR.append(0), mP.append(0)
                continue

            # Get detections sorted by decreasing confidence scores
            detections = detections.cpu().numpy()
            detections = detections[np.argsort(-detections[:, 4])]

            # If no labels add number of detections as incorrect
            correct = []
            if labels.size(0) == 0:
                # correct.extend([0 for _ in range(len(detections))])
                mAPs.append(0), mR.append(0), mP.append(0)
                continue
            else:
                target_cls = labels[:, 0]

                # Extract target boxes as (x1, y1, x2, y2)
                target_boxes = xywh2xyxy(labels[:, 1:5]) * img_size

                detected = []
                for *pred_bbox, conf, obj_conf, obj_pred in detections:

                    pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                    # Compute iou with target boxes
                    iou = bbox_iou(pred_bbox, target_boxes)
                    # Extract index of largest overlap
                    best_i = np.argmax(iou)
                    # If overlap exceeds threshold and classification is correct mark as correct
                    if iou[best_i] > iou_thres and obj_pred == labels[best_i, 0] and best_i not in detected:
                        correct.append(1)
                        detected.append(best_i)
                    else:
                        correct.append(0)

            # Compute Average Precision (AP) per class
            AP, AP_class, R, P = ap_per_class(tp=correct,
                                              conf=detections[:, 4],
                                              pred_cls=detections[:, 6],
                                              target_cls=target_cls)

            # Accumulate AP per class
            AP_accum_count += np.bincount(AP_class, minlength=nC)
            AP_accum += np.bincount(AP_class, minlength=nC, weights=AP)

            # Compute mean AP across all classes in this image, and append to image list
            mAPs.append(AP.mean())
            mR.append(R.mean())
            mP.append(P.mean())

            # Means of all images
            mean_mAP = np.mean(mAPs)
            mean_R = np.mean(mR)
            mean_P = np.mean(mP)

        # Print image mAP and running mean mAP
        print(('%11s%11s' + '%11.3g' * 3) % (seen, dataloader.nF, mean_P, mean_R, mean_mAP))

    # Print mAP per class
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP') + '\n\nmAP Per Class:')

    classes = load_classes(data_cfg_dict['names'])  # Extracts class labels from file
    for i, c in enumerate(classes):
        print('%15s: %-.4f' % (c, AP_accum[i] / (AP_accum_count[i] + 1E-16)))

    # Return mAP
    return mean_mAP, mean_R, mean_P


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='cfg/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    with torch.no_grad():
        mAP = test(
            opt.cfg,
            opt.data_cfg,
            opt.weights,
            opt.batch_size,
            opt.img_size,
            opt.iou_thres,
            opt.conf_thres,
            opt.nms_thres
        )

#       Image      Total          P          R        mAP  # YOLOv3 320
#          32       5000       0.66      0.597      0.591
#          64       5000      0.664       0.62      0.604
#          96       5000      0.653      0.627      0.614
#         128       5000      0.639      0.623      0.607
#         160       5000      0.642       0.63      0.616
#         192       5000      0.651      0.636      0.621

#       Image      Total          P          R        mAP  # YOLOv3 416
#          32       5000      0.635      0.581       0.57
#          64       5000       0.63      0.591      0.578
#          96       5000      0.661      0.632      0.622
#         128       5000      0.659      0.632      0.623
#         160       5000      0.665       0.64      0.633
#         192       5000       0.66      0.637       0.63

#       Image      Total          P          R        mAP  # YOLOv3 608
#          32       5000      0.653      0.606      0.591
#          64       5000      0.653      0.635      0.625
#          96       5000      0.655      0.642      0.633
#         128       5000      0.667      0.651      0.642
#         160       5000      0.663      0.645      0.637
#         192       5000      0.663      0.643      0.634
