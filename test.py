import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *


def test(
        cfg,
        data_cfg,
        weights=None,
        batch_size=16,
        img_size=416,
        iou_thres=0.5,
        conf_thres=0.1,
        nms_thres=0.5,
        save_json=False,
        model=None
):
    if model is None:
        device = torch_utils.select_device()

        # Initialize model
        model = Darknet(cfg, img_size).to(device)

        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device

    # Configure run
    data_cfg = parse_data_cfg(data_cfg)
    test_path = data_cfg['valid']
    if (os.sep + 'coco' + os.sep) in test_path:  # COCO dataset probable
        save_json = True  # use pycocotools

    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size=img_size)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=4,
                            pin_memory=False,
                            collate_fn=dataset.collate_fn)

    model.eval()
    seen = 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    mP, mR, mAP, mAPj = 0.0, 0.0, 0.0, 0.0
    jdict, tdict, stats, AP, AP_class = [], [], [], [], []
    coco91class = coco80_to_coco91_class()
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc='Calculating mAP')):
        targets = targets.to(device)
        imgs = imgs.to(device)

        output = model(imgs)
        output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)

        # Per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            correct, detected = [], []
            tcls = torch.Tensor()
            seen += 1

            if pred is None:
                continue

            if save_json:  # add to json pred dictionary
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img_size, box, shapes[si])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for di, d in enumerate(pred):
                    jdict.append({
                        'image_id': image_id,
                        'category_id': coco91class[int(d[6])],
                        'bbox': [float3(x) for x in box[di]],
                        'score': float(d[4])
                    })

            if len(labels):
                # Extract target boxes as (x1, y1, x2, y2)
                tbox = xywh2xyxy(labels[:, 1:5]) * img_size  # target boxes
                tcls = labels[:, 0]  # target classes

                for *pbox, pconf, pcls_conf, pcls in pred:
                    if pcls not in tcls:
                        correct.append(0)
                        continue

                    # Best iou, index between pred and targets
                    iou, bi = bbox_iou(pbox, tbox).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and bi not in detected:
                        correct.append(1)
                        detected.append(bi)
                    else:
                        correct.append(0)
            else:
                # If no labels add number of detections as incorrect
                correct.extend([0] * len(pred))

            # Append Statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls.cpu()))

    # Compute means
    stats_np = [np.concatenate(x, 0) for x in list(zip(*stats))]
    if len(stats_np):
        AP, AP_class, R, P = ap_per_class(*stats_np)
        mP, mR, mAP = P.mean(), R.mean(), AP.mean()

    # Print P, R, mAP
    print(('%11s%11s' + '%11.3g' * 3) % (seen, len(dataset), mP, mR, mAP))

    # Print mAP per class
    if len(stats_np):
        print('\nmAP Per Class:')
        names = load_classes(data_cfg['names'])
        for c, a in zip(AP_class, AP):
            print('%15s: %-.4f' % (names[c], a))

    # Save JSON
    if save_json and mAP and len(jdict):
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataset.img_files]
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        cocoGt = COCO('../coco/annotations/instances_val2014.json')  # initialize COCO ground truth api
        cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        mAP = cocoEval.stats[1]  # update mAP to pycocotools mAP

    # F1 score = harmonic mean of precision and recall
    # F1 = 2 * (mP * mR) / (mP + mR)

    # Return mAP
    return mP, mR, mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='cfg/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
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
            opt.nms_thres,
            opt.save_json
        )
