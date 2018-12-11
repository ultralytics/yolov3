import argparse
import time

from models import *
from utils.datasets import *
from utils.utils import *

from utils import torch_utils


def detect(
        net_config_path,
        data_config_path,
        images_path,
        weights='weights/yolov3.pt',
        output='output',
        batch_size=16,
        img_size=416,
        conf_thres=0.3,
        nms_thres=0.45,
        save_txt=False,
        save_images=False,
):
    device = torch_utils.select_device()
    print("Using device: \"{}\"".format(device))

    os.system('rm -rf ' + output)
    os.makedirs(output, exist_ok=True)

    data_config = parse_data_config(data_config_path)

    # Load model
    model = Darknet(net_config_path, img_size)

    if weights.endswith('.pt'):  # pytorch format
        if weights.endswith('weights/yolov3.pt') and not os.path.isfile(weights):
            os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)
        checkpoint = torch.load(weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        del checkpoint
    else:  # darknet format
        load_weights(model, weights)

        # current = model.state_dict()
        # saved = checkpoint['model']
        # # 1. filter out unnecessary keys
        # saved = {k: v for k, v in saved.items() if ((k in current) and (current[k].shape == v.shape))}
        # # 2. overwrite entries in the existing state dict
        # current.update(saved)
        # # 3. load the new state dict
        # model.load_state_dict(current)
        # model.to(device).eval()
        # del checkpoint, current, saved

    model.to(device).eval()

    # Set Dataloader
    classes = load_classes(data_config['names'])  # Extracts class labels from file
    dataloader = load_images(images_path, batch_size=batch_size, img_size=img_size)

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    prev_time = time.time()
    for i, (img_paths, img) in enumerate(dataloader):
        print('%g/%g' % (i + 1, len(dataloader)), end=' ')

        # Get detections
        with torch.no_grad():
            pred = model(torch.from_numpy(img).unsqueeze(0).to(device))
            pred = pred[pred[:, :, 4] > conf_thres]

            if len(pred) > 0:
                detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)
                img_detections.extend(detections)
                imgs.extend(img_paths)

        print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
        prev_time = time.time()

    # Bounding-box colors
    color_list = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

    if len(img_detections) == 0:
        return

    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("image %g: '%s'" % (img_i, path))

        if save_images:
            img = cv2.imread(path)

        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        # Image height and width after padding is removed
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x

        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_classes = detections[:, -1].cpu().unique()
            bbox_colors = random.sample(color_list, len(unique_classes))

            # write results to .txt file
            results_img_path = os.path.join(output, path.split('/')[-1])
            results_txt_path = results_img_path + '.txt'
            if os.path.isfile(results_txt_path):
                os.remove(results_txt_path)

            for i in unique_classes:
                n = (detections[:, -1].cpu() == i).sum()
                print('%g %ss' % (n, classes[int(i)]))

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = (((y1 - pad_y // 2) / unpad_h) * img.shape[0]).round().item()
                x1 = (((x1 - pad_x // 2) / unpad_w) * img.shape[1]).round().item()
                x2 = (x1 + box_w).round().item()
                y2 = (y1 + box_h).round().item()
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)

                # write to file
                if save_txt:
                    with open(results_txt_path, 'a') as file:
                        file.write(('%g %g %g %g %g %g \n') % (x1, y1, x2, y2, cls_pred, cls_conf * conf))

                if save_images:
                    # Add the bbox to the plot
                    label = '%s %.2f' % (classes[int(cls_pred)], conf)
                    color = bbox_colors[int(np.where(unique_classes == int(cls_pred))[0])]
                    plot_one_box([x1, y1, x2, y2], img, label=label, color=color)

        if save_images:
            # Save generated image with detections
            cv2.imwrite(results_img_path.replace('.bmp', '.jpg').replace('.tif', '.jpg'), img)

    if platform == 'darwin':  # MacOS (local)
        os.system('open ' + output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Get data configuration

    parser.add_argument('images', type=str, help='path to images')

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

    algorithm_group = parser.add_argument_group(
        title='experiment arguments',
        description='Configure the experiment'
    )
    algorithm_group.add_argument('--batch', type=int, default=32, help='size of the batches')
    algorithm_group.add_argument('--thr', type=float, default=0.3, help='object confidence threshold')
    algorithm_group.add_argument('--nms', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    algorithm_group.add_argument('--output', type=str, default='output', help='path to outputs')
    algorithm_group.add_argument('--save-txt', action='store_true')
    algorithm_group.add_argument('--save-images', action='store_true')

    opt = parser.parse_args()
    print(opt)

    torch.cuda.empty_cache()

    init_seeds()

    detect(
        opt.cfg,
        opt.data,
        opt.images,
        weights=opt.weights,
        output=opt.output,
        batch_size=opt.batch,
        img_size=opt.img_size,
        conf_thres=opt.thr,
        nms_thres=opt.nms,
        save_txt=opt.save_txt,
        save_images=opt.save_images,
    )
