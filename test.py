import argparse

from models import *
from utils.datasets import *
from utils.utils import *

from utils import torch_utils


def test(
        net_config_path,
        data_config_path,
        weights_file_path,
        batch_size=16,
        img_size=416,
        iou_thres=0.5,
        conf_thres=0.3,
        nms_thres=0.45,
):
    device = torch_utils.select_device()
    print("Using device: \"{}\"".format(device))

    # Configure run
    data_config = parse_data_config(data_config_path)
    nC = int(data_config['classes'])  # number of classes (80 for COCO)
    test_path = data_config['valid']

    # Initiate model
    model = Darknet(net_config_path, img_size)

    # Load weights
    if weights_file_path.endswith('.pt'):  # pytorch format
        checkpoint = torch.load(weights_file_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        del checkpoint
    else:  # darknet format
        load_weights(model, weights_file_path)

    model.to(device).eval()

    # Get dataloader
    # dataset = load_images_with_labels(test_path)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpus)
    dataloader = load_images_and_labels(test_path, batch_size=batch_size, img_size=img_size)

    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class = [], [], [], [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)
    for batch_i, (imgs, targets) in enumerate(dataloader):

        with torch.no_grad():
            output = model(imgs.to(device))
            output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)

        # Compute average precision for each sample
        for sample_i, (labels, detections) in enumerate(zip(targets, output)):
            correct = []

            if detections is None:
                # If there are no detections but there are labels mask as zero AP
                if labels.size(0) != 0:
                    mAPs.append(0), mR.append(0), mP.append(0)
                continue

            # Get detections sorted by decreasing confidence scores
            detections = detections.cpu().numpy()
            detections = detections[np.argsort(-detections[:, 4])]

            # If no labels add number of detections as incorrect
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
            AP, AP_class, R, P = ap_per_class(tp=correct, conf=detections[:, 4], pred_cls=detections[:, 6],
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
            print(('%11s%11s' + '%11.3g' * 3) % (len(mAPs), dataloader.nF, mean_P, mean_R, mean_mAP))

    # Print mAP per class
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP') + '\n\nmAP Per Class:')

    classes = load_classes(data_config['names'])  # Extracts class labels from file
    for i, c in enumerate(classes):
        print('%15s: %-.4f' % (c, AP_accum[i] / AP_accum_count[i]))

    # Return mAP
    return mean_mAP, mean_R, mean_P


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Get network configuration
    network_group = parser.add_argument_group(
        title='network arguments',
        description='Configure network and data'
    )
    network_group.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    network_group.add_argument('--weights', type=str, default='weights/yolov3.pt', help='path to weights file')

    # Get data configuration
    data_group = parser.add_argument_group(
        title='data arguments',
        description='Configure data'
    )
    data_group.add_argument('--data', type=str, default='cfg/coco.data', help='path to data config file')
    data_group.add_argument('--img-size', type=int, default=32 * 13, help='size of each image in pixels')
    data_group.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')

    algorithm_group = parser.add_argument_group(
        title='experiment arguments',
        description='Configure the experiment'
    )
    algorithm_group.add_argument('--batch', type=int, default=32, help='size of the batches')
    algorithm_group.add_argument('--thr', type=float, default=0.3, help='object confidence threshold')
    algorithm_group.add_argument('--nms', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    algorithm_group.add_argument('--iou', type=float, default=0.5, help='iou threshold required to qualify as detected')
    algorithm_group.add_argument('--resume', action='store_true', help='resume training flag')
    algorithm_group.add_argument('--report', action='store_true', help='report TP, FP, FN, P and R per batch (slower)')

    opt = parser.parse_args()

    print(opt, end='\n\n')

    init_seeds()

    mAP = test(
        opt.cfg,
        opt.data,
        opt.weights,
        batch_size=opt.batch,
        img_size=opt.img_size,
        iou_thres=opt.iou,
        conf_thres=opt.thr,
        nms_thres=opt.nms,
    )
