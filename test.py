import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *


def test(cfg,
         data,
         weights=None,
         batch_size=16,
         img_size=416,
         conf_thres=0.001,
         iou_thres=0.5,  # for nms
         save_json=False,
         single_cls=False,
         model=None,
         dataloader=None):
    # Initialize/load model and set device
    if model is None:
        device = torch_utils.select_device(opt.device, batch_size=batch_size)
        verbose = opt.task == 'test'

        # Remove previous
        for f in glob.glob('test_batch*.jpg'):
            os.remove(f)

        # Initialize model
        model = Darknet(cfg, img_size).to(device)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            load_darknet_weights(model, weights)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:  # called by train.py
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    data = parse_data_cfg(data)
    nc = 1 if single_cls else int(data['classes'])  # number of classes
    path = data['valid']  # path to test images
    names = load_classes(data['names'])  # class names
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(path, img_size, batch_size, rect=True)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        # Plot images with bounding boxes
        if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
            plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.jpg')

        # Disable gradients
        with torch.no_grad():
            # Run model
            inf_out, train_out = model(imgs)  # inference and training outputs

            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3].cpu()  # GIoU, obj, cls

            # Run NMS
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for di, d in enumerate(pred):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(d[5])],
                                  'bbox': [floatn(x, 3) for x in box[di]],
                                  'score': floatn(d[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(len(pred), niou, dtype=torch.bool)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * torch.Tensor([width, height, width, height]).to(device)

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

                    # Search for detections
                    if len(pi):
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = (ious[j] > iouv).cpu()  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Save JSON
    if save_json and map and len(jdict):
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except:
            print('WARNING: missing pycocotools package, can not compute official COCO mAP. See requirements.txt.')

        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
        cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist()), maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2014.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    opt = parser.parse_args()
    opt.save_json = opt.save_json or any([x in opt.data for x in ['coco.data', 'coco2014.data', 'coco2017.data']])
    print(opt)

    if opt.task == 'test':  # task = 'test', 'study', 'benchmark'
        # Test
        test(opt.cfg,
             opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls)

    elif opt.task == 'benchmark':
        # mAPs at 320-608 at conf 0.5 and 0.7
        y = []
        for i in [320, 416, 512, 608]:
            for j in [0.5, 0.7]:
                t = time.time()
                r = test(opt.cfg, opt.data, opt.weights, opt.batch_size, i, opt.conf_thres, j, opt.save_json)[0]
                y.append(r + (time.time() - t,))
        np.savetxt('benchmark.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')

    elif opt.task == 'study':
        # Parameter study
        y = []
        x = np.arange(0.4, 0.9, 0.05)
        for i in x:
            t = time.time()
            r = test(opt.cfg, opt.data, opt.weights, opt.batch_size, opt.img_size, opt.conf_thres, i, opt.save_json)[0]
            y.append(r + (time.time() - t,))
        np.savetxt('study.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')

        # Plot
        fig, ax = plt.subplots(3, 1, figsize=(6, 6))
        y = np.stack(y, 0)
        ax[0].plot(x, y[:, 2], marker='.', label='mAP@0.5')
        ax[0].set_ylabel('mAP')
        ax[1].plot(x, y[:, 3], marker='.', label='mAP@0.5:0.95')
        ax[1].set_ylabel('mAP')
        ax[2].plot(x, y[:, -1], marker='.', label='time')
        ax[2].set_ylabel('time (s)')
        for i in range(3):
            ax[i].legend()
            ax[i].set_xlabel('iou_thr')
        fig.tight_layout()
        plt.savefig('study.jpg', dpi=200)
