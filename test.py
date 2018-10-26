import argparse

from models import *
from utils.datasets import *
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', type=int, default=32, help='size of each image batch')
parser.add_argument('-cfg', type=str, default='cfg/yolov3.cfg', help='path to model config file')
parser.add_argument('-data_config_path', type=str, default='cfg/coco.data', help='path to data config file')
parser.add_argument('-weights_path', type=str, default='weights/yolov3.pt', help='path to weights file')
parser.add_argument('-class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('-iou_thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
parser.add_argument('-conf_thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
parser.add_argument('-n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('-img_size', type=int, default=416, help='size of each image dimension')
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

# Configure run
data_config = parse_data_config(opt.data_config_path)
num_classes = int(data_config['classes'])
if platform == 'darwin':  # MacOS (local)
    test_path = data_config['valid']
else:  # linux (cloud, i.e. gcp)
    test_path = '../coco/5k.part'

# Initiate model
model = Darknet(opt.cfg, opt.img_size)

# Load weights
if opt.weights_path.endswith('.weights'):  # darknet format
    load_weights(model, opt.weights_path)
elif opt.weights_path.endswith('.pt'):  # pytorch format
    checkpoint = torch.load(opt.weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    del checkpoint

model.to(device).eval()

# Get dataloader
# dataset = load_images_with_labels(test_path)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
dataloader = load_images_and_labels(test_path, batch_size=opt.batch_size, img_size=opt.img_size)

print('Compute mAP...')

nC = 80  # number of classes
correct = 0
targets = None
outputs, mAPs, TP, confidence, pred_class, target_class = [], [], [], [], [], []
AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)
for batch_i, (imgs, targets) in enumerate(dataloader):
    imgs = imgs.to(device)

    with torch.no_grad():
        output = model(imgs)
        output = non_max_suppression(output, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

    # Compute average precision for each sample
    for sample_i in range(len(targets)):
        correct = []

        # Get labels for sample where width is not zero (dummies)
        annotations = targets[sample_i]
        # Extract detections
        detections = output[sample_i]

        if detections is None:
            # If there are no detections but there are annotations mask as zero AP
            if annotations.size(0) != 0:
                mAPs.append(0)
            continue

        # Get detections sorted by decreasing confidence scores
        detections = detections[np.argsort(-detections[:, 4])]

        # If no annotations add number of detections as incorrect
        if annotations.size(0) == 0:
            target_cls = []
            # correct.extend([0 for _ in range(len(detections))])
            mAPs.append(0)
            continue
        else:
            target_cls = annotations[:, 0]

            # Extract target boxes as (x1, y1, x2, y2)
            target_boxes = xywh2xyxy(annotations[:, 1:5])
            target_boxes *= opt.img_size

            detected = []
            for *pred_bbox, conf, obj_conf, obj_pred in detections:

                pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                # Compute iou with target boxes
                iou = bbox_iou(pred_bbox, target_boxes)
                # Extract index of largest overlap
                best_i = np.argmax(iou)
                # If overlap exceeds threshold and classification is correct mark as correct
                if iou[best_i] > opt.iou_thres and obj_pred == annotations[best_i, 0] and best_i not in detected:
                    correct.append(1)
                    detected.append(best_i)
                else:
                    correct.append(0)

        # Compute Average Precision (AP) per class
        AP, AP_class = ap_per_class(tp=correct, conf=detections[:, 4], pred_cls=detections[:, 6], target_cls=target_cls)

        # Accumulate AP per class
        AP_accum_count += np.bincount(AP_class, minlength=nC)
        AP_accum += np.bincount(AP_class, minlength=nC, weights=AP)

        # Compute mean AP for this image
        mAP = AP.mean()

        # Append image mAP to list
        mAPs.append(mAP)

        # Print image mAP and running mean mAP
        print('+ Sample [%d/%d] AP: %.4f (%.4f)' % (len(mAPs), len(dataloader) * opt.batch_size, mAP, np.mean(mAPs)))

# Print mAP per class
classes = load_classes(opt.class_path)  # Extracts class labels from file
for i, c in enumerate(classes):
    print('%15s: %-.4f' % (c, AP_accum[i] / AP_accum_count[i]))

# Print mAP
print('Mean Average Precision: %.4f' % np.mean(mAPs))
