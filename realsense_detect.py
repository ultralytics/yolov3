import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import cv2
import pyrealsense2 as rs
import torchvision.transforms as transforms

def detect(pipe = None, save_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()
    # torch_utils.model_info(model, report='summary')  # 'full' or 'summary'

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=False, opset_version=11)

        # Validate exported model
        import onnx
        model = onnx.load('weights/export.onnx')  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        #dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        save_img = True
        #dataset = LoadImages(source, img_size=img_size, half=half)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    # for path, img, im0s, vid_cap in dataset:
    while True:
        t = time.time()

        # Get detections
        img = pipe.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = align.process(img)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        aligned_color_frame = aligned_frames.get_color_frame()

        aligned_color_frame = np.asanyarray(img.get_data())
        #im0 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #im0 = cv2.resize(im0, (640,480))
        aligned_color_frame = cv2.resize(img, (320, 320))
        im0 = cv2.cvtColor(aligned_color_frame, cv2.COLOR_RGB2BGR)       # im0 is original image
        img = transforms.ToTensor()(im0).to(device)
        
        aligned_depth_frame = cv2.resize(aligned_depth_frame, (320, 320))

        #img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        #if classify:
        #    pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # depth mask for each of the objects
            class_mask = np.zeros((320, 320, 21))

            s = ""
            #if webcam:  # batch_size >= 1
            #    p, s, im0 = " ", '%g: ' % i, im0s[i]
            #else:
            #    p, s, im0 = path, '', im0s

            # save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
            #print(det)
            
            # fill in bounding box area for mask
            # cv2.contour for each channel
            # for each contour -> find the centroid

            # depth + color image combine to overlay 


            # Stream results
            if True:
                print(1 / (time.time() - t))
                #cv2.imshow("output", im0)
                #cv2.imshow("resized", cv2.resize(im0, (1080,720)))
                
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            #if save_img:
                #if dataset.mode == 'images':
                #    cv2.imwrite(save_path, im0)
                #else:
                #    if vid_path != save_path:  # new video
                #        vid_path = save_path
                #        if isinstance(vid_writer, cv2.VideoWriter):
                #            vid_writer.release()  # release previous video writer

                 #       fps = vid_cap.get(cv2.CAP_PROP_FPS)
                 #       w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                 #       h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                 #       vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                  #  vid_writer.write(im0)

    #if save_txt or save_img:
    #    print('Results saved to %s' % os.getcwd() + os.sep + out)
    #    if platform == 'darwin':  # MacOS
    #        os.system('open ' + out + ' ' + save_path)

    #print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)

    align_to = rs.stream.color
    align = rs.align(align_to)

    pipeline = rs.pipeline()
    pipeline.start()

    with torch.no_grad():
        detect(pipe = pipeline)
