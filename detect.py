import argparse
import time

from models import *
from utils.datasets import *
from utils.utils import *

from utils import torch_utils


def detect(cfg, weights, images, output='output', img_size=416, conf_thres=0.3, nms_thres=0.45,
           save_txt=False, save_images=True):
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

    # Classes and colors
    classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])  # Extracts class labels from file
    colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

    for i, (path, img, img0) in enumerate(dataloader):
        print("%g/%g '%s': " % (i + 1, len(dataloader), path), end='')
        t = time.time()

        # Get detections
        with torch.no_grad():
            img = torch.from_numpy(img).unsqueeze(0).to(device)
            if ONNX_EXPORT:
                pred = torch.onnx._export(model, img, 'weights/model.onnx', verbose=True)
                return  # ONNX export
            pred = model(img)
            pred = pred[pred[:, :, 4] > conf_thres]

            if len(pred) > 0:
                detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]

            # Draw bounding boxes and labels of detections
            if detections is not None:
                save_img_path = os.path.join(output, path.split('/')[-1])
                save_txt_path = save_img_path + '.txt'
                img = img0

                # The amount of padding that was added
                pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
                pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
                # Image height and width after padding is removed
                unpad_h = img_size - pad_y
                unpad_w = img_size - pad_x

                unique_classes = detections[:, -1].cpu().unique()
                for i in unique_classes:
                    n = (detections[:, -1].cpu() == i).sum()
                    print('%g %ss' % (n, classes[int(i)]), end=', ')

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    # Rescale coordinates to original dimensions
                    box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                    box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                    y1 = (((y1 - pad_y // 2) / unpad_h) * img.shape[0]).round()
                    x1 = (((x1 - pad_x // 2) / unpad_w) * img.shape[1]).round()
                    x2 = (x1 + box_w).round()
                    y2 = (y1 + box_h).round()
                    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)

                    # write to file
                    if save_txt:
                        with open(save_txt_path, 'a') as file:
                            file.write(('%g %g %g %g %g %g\n') % (x1, y1, x2, y2, cls_pred, cls_conf * conf))

                    if save_images:
                        # Add bbox to the image
                        label = '%s %.2f' % (classes[int(cls_pred)], conf)
                        plot_one_box([x1, y1, x2, y2], img, label=label, color=colors[int(cls_pred)])

                if save_images:
                    # Save generated image with detections
                    cv2.imwrite(save_img_path, img)

        print(' Done. (%.3fs)' % (time.time() - t))

    if platform == 'darwin':  # MacOS
        os.system('open ' + output)
        os.system('open ' + save_img_path)



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

    detect(opt.cfg, opt.weights, opt.images, img_size=opt.img_size, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)
