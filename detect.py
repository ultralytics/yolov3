import argparse
import time

from models import *
from utils.datasets import *
from utils.utils import *

from utils import torch_utils


def detect(
        cfg,
        weights,
        images,
        output='output',
        img_size=416,
        conf_thres=0.3,
        nms_thres=0.45,
        save_txt=False,
        save_images=True
):
    device = torch_utils.select_device()
    os.system('rm -rf ' + output)
    os.makedirs(output, exist_ok=True)

    # Load model
    model = Darknet(cfg, img_size)

    if weights.endswith('.pt'):  # pytorch format
        if weights.endswith('weights/yolov3.pt') and not os.path.isfile(weights):
            os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    model.to(device).eval()

    # Set Dataloader
    dataloader = load_images(images, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])
    colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

    for i, (path, img, im0) in enumerate(dataloader):
        print("%g/%g '%s': " % (i + 1, len(dataloader), path), end='')
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx._export(model, img, 'weights/model.onnx', verbose=True)
            return  # ONNX export
        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_thres]

        if len(pred) > 0:
            detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]

        # Draw bounding boxes and labels of detections
        if detections is not None:
            save_path = os.path.join(output, path.split('/')[-1])

            # Rescale boxes from 416 to true image size
            detections[:, :4] = scale_coords(img_size, detections[:, :4], im0.shape)

            unique_classes = detections[:, -1].cpu().unique()
            for i in unique_classes:
                n = (detections[:, -1].cpu() == i).sum()
                print('%g %ss' % (n, classes[int(i)]), end=', ')

            for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write('%g %g %g %g %g %g\n' % (x1, y1, x2, y2, cls, cls_conf * conf))

                if save_images:  # Add bbox to the image
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box([x1, y1, x2, y2], im0, label=label, color=colors[int(cls)])

            if save_images:  # Save generated image with detections
                cv2.imwrite(save_path, im0)

        print(' Done. (%.3fs)' % (time.time() - t))

    if platform == 'darwin':  # MacOS
        os.system('open ' + output + '&& open ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres
        )
