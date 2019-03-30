import argparse
import time
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *


def detect(
        cfg,
        weights,
        images,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.3,
        nms_thres=0.5,
        save_txt=False,
        save_images=True,
        webcam=False
):
    device = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    model.to(device).eval()

    # Set Dataloader
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    for i, (path, img, im0) in enumerate(dataloader):
        t = time.time()
        save_path = str(Path(output) / Path(path).name)
        if webcam:
            print('webcam frame %g: ' % (i + 1), end='')
        else:
            print('image %g/%g %s: ' % (i + 1, len(dataloader), path), end='')

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return
        pred = model(img)
        detections = non_max_suppression(pred, conf_thres, nms_thres)[0]

        if len(detections) > 0:
            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()

            # Print results to screen
            for c in detections[:, -1].unique():
                n = (detections[:, -1] == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in detections:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        print('Done. (%.3fs)' % (time.time() - t))

        if save_images:  # Save generated image with detections
            cv2.imwrite(save_path, im0)

        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)

    if save_images and platform == 'darwin':  # macos
        os.system('open ' + output + ' ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
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
