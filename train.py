import argparse
import time

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed

# 320 --epochs 1
#      0.109      0.297       0.150       0.126       7.04      1.666      4.062     0.1845       42.6       3.34      12.61      8.338     0.2705      0.001         -4        0.9     0.0005 a  320 giou + best_anchor False
#      0.223      0.218       0.138       0.189       9.28      1.153      4.376    0.08263      24.28       3.05      20.93      2.842     0.2759   0.001357     -5.036     0.9158  0.0005722 b  mAP/F1 - 50/50 weighting
#      0.231      0.215       0.135       0.191       9.51      1.432      3.007    0.06082      24.87      3.477      24.13      2.802     0.3436   0.001127     -5.036     0.9232  0.0005874 c
#      0.246      0.194       0.128       0.192       8.12      1.101      3.954     0.0817      22.83      3.967      19.83      1.779     0.3352   0.000895     -5.036     0.9238  0.0007973 d
#      0.187      0.237       0.144       0.186       14.6      1.607      4.202    0.09439      39.27      3.726      31.26      2.634      0.273   0.001542     -5.036     0.8364  0.0008393 e
#      0.250      0.217       0.136       0.195         3.3         1.2           2       0.604        15.7        3.67          20        1.36       0.194     0.00128          -4        0.95    0.000201         0.8       0.388         1.2       0.119      0.0589       0.401 f
#      0.269      0.225       0.149       0.218        6.71        1.13        5.25       0.246        22.4        3.64        17.8        1.31       0.256     0.00146          -4       0.936     0.00042       0.123        0.18        1.81      0.0987      0.0788       0.441 g
#      0.179      0.274       0.165       0.187        7.95        1.22        7.62       0.224          17        5.71        17.7        3.28       0.295     0.00136          -4       0.875    0.000319       0.131       0.208        2.14        0.14      0.0773       0.228 h
#      0.296      0.228       0.152       0.220        5.18        1.43        4.27       0.265        11.7        4.81        11.5        1.56       0.281      0.0013          -4       0.944    0.000427      0.0599       0.142        1.03      0.0552      0.0555       0.434 i

# 320 --epochs 2
# 0.242	0.296	0.196	0.231	5.67	0.8541	4.286	0.1539	21.61	1.957	22.9	2.894	0.3689	0.001844	-4	0.913	0.000467  # ha 0.417 mAP @ epoch 100
# 0.298	0.244	0.167	0.247	4.99	0.8896	4.067	0.1694	21.41	2.033	25.61	1.783	0.4115	0.00128	    -4	0.950	0.000377  # hb
# 0.268	0.268	0.178	0.240	4.36	1.104	5.596	0.2087	14.47	2.599	16.27	2.406	0.4114	0.001585	-4	0.950	0.000524  # hc
# 0.161	0.327	0.190	0.193	7.82	1.153	4.062	0.1845	24.28	3.05	20.93	2.842	0.2759	0.001357	-4	0.916	0.000572  # hd 0.438 mAP @ epoch 100

# Hyperparameters (j-series, 50.5 mAP yolov3-320) evolved by @ktian08 https://github.com/ultralytics/yolov3/issues/310
# Transfer learning edge layers: 0.1 lr0, 0.9 momentum
hyp = {'giou': 1.582,  # giou loss gain
       'xy': 4.688,  # xy loss gain
       'wh': 0.1857,  # wh loss gain
       'cls': 27.76,  # cls loss gain  (CE=~1.0, uCE=~20)
       'cls_pw': 1.446,  # cls BCELoss positive_weight
       'obj': 21.35,  # obj loss gain (*=80 for uBCE with 80 classes)
       'obj_pw': 3.941,  # obj BCELoss positive_weight
       'iou_t': 0.2635,  # iou training threshold
       'lr0': 0.002324,  # initial learning rate
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.97,  # SGD momentum
       'weight_decay': 0.0004569,  # optimizer weight decay
       'hsv_s': 0.5703,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.3174,  # image HSV-Value augmentation (fraction)
       'degrees': 1.113,  # image rotation (+/- deg)
       'translate': 0.06797,  # image translation (+/- fraction)
       'scale': 0.1059,  # image scale (+/- gain)
       'shear': 0.5768}  # image shear (+/- deg)


def train():
    cfg = opt.cfg
    data = opt.data
    img_size = opt.img_size
    epochs = opt.epochs  # 500200 batches at bs 16, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights  # initial training weights

    # Initialize
    init_seeds()
    wdir = 'weights' + os.sep  # weights dir
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'
    device = torch_utils.select_device(apex=mixed_precision)
    multi_scale = opt.multi_scale

    if multi_scale:
        img_sz_min = round(img_size / 32 / 1.5) + 1
        img_sz_max = round(img_size / 32 * 1.5) - 1
        img_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))

    # Configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    nc = int(data_dict['classes'])  # number of classes

    # Initialize model
    model = Darknet(cfg, arc=opt.arc).to(device)

    # Optimizer
    # optimizer = optim.Adam(model.parameters(), lr=hyp['lr0'], weight_decay=hyp['weight_decay'])
    # optimizer = AdaBound(model.parameters(), lr=hyp['lr0'], final_lr=0.1)
    optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'],
                          nesterov=True)

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_fitness = 0.
    if weights.endswith('.pt'):  # pytorch format
        # possible weights are 'last.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        if opt.bucket:
            os.system('gsutil cp gs://%s/last.pt %s' % (opt.bucket, last))  # download from bucket
        chkpt = torch.load(weights, map_location=device)

        # load model
        if opt.transfer:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
        else:
            model.load_state_dict(chkpt['model'])

        # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        # load results
        if chkpt.get('training_results') is not None:
            with open('results.txt', 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt

    elif len(weights) > 0:  # darknet format
        # possible weights are 'yolov3.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        cutoff = load_darknet_weights(model, weights)

    if opt.transfer:  # transfer learning
        nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)

        for x in optimizer.param_groups:  # lower parameter count can handle more aggressive training hyps
            x['lr'] *= 10
            x['momentum'] *= 0.9

        for p in model.parameters():
            p.requires_grad = True if p.shape[0] == nf else False

    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    # lf = lambda x: 1 - x / epochs  # linear ramp to zero
    # lf = lambda x: 10 ** (hyp['lrf'] * x / epochs)  # exp ramp
    # lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inverse exp ramp
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    # # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  image_weights=opt.img_weights,
                                  cache_images=opt.cache_images)

    # Dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=min(os.cpu_count(), batch_size),
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob('results.txt'):
        os.remove(f)

    # Start training
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model_info(model, report='summary')  # 'full' or 'summary'
    nb = len(dataloader)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    t0 = time.time()
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------

        model.train()
        print(('\n' + '%10s' * 9) %
              ('Epoch', 'gpu_mem', 'GIoU/xy', 'wh', 'obj', 'cls', 'total', 'targets', 'img_size'))

        # Update scheduler
        if epoch > 0:
            scheduler.step()

        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
        freeze_backbone = False
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(5).to(device)  # mean losses
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Multi-Scale training
            if multi_scale:
                if ni / accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                    img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Plot images with bounding boxes
            if ni == 0:
                fname = 'train_batch%g.jpg' % i
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=fname)
                if tb_writer:
                    tb_writer.add_image(fname, cv2.imread(fname)[:, :, ::-1], dataformats='HWC')

            # Hyperparameter burn-in
            # n_burn = nb - 1  # min(nb // 5 + 1, 1000)  # number of burn-in batches
            # if ni <= n_burn:
            #     for m in model.named_modules():
            #         if m[0].endswith('BatchNorm2d'):
            #             m[1].momentum = 1 - i / n_burn * 0.99  # BatchNorm2d momentum falls from 1 - 0.01
            #     g = (i / n_burn) ** 4  # gain rises from 0 - 1
            #     for x in optimizer.param_groups:
            #         x['lr'] = hyp['lr0'] * g
            #         x['weight_decay'] = hyp['weight_decay'] * g

            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model, arc=opt.arc)
            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
                return results

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Accumulate gradient for x batches before optimizing
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

        # Calculate mAP (always test final epoch, skip first 5 if opt.nosave)
        final_epoch = epoch + 1 == epochs
        if not (opt.notest or (opt.nosave and epoch < 10)) or final_epoch:
            with torch.no_grad():
                results, maps = test.test(cfg,
                                          data,
                                          batch_size=batch_size,
                                          img_size=opt.img_size,
                                          model=model,
                                          conf_thres=0.001 if final_epoch and epoch > 0 else 0.1,  # 0.1 for speed
                                          save_json=final_epoch and epoch > 0 and 'coco.data' in data)

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        # Write Tensorboard results
        if tb_writer:
            x = list(mloss[:5]) + list(results[:7])
            titles = ['GIoU/XY', 'Width/Height', 'Objectness', 'Classification', 'Train loss', 'Precision', 'Recall',
                      'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification']
            for xi, title in zip(x, titles):
                tb_writer.add_scalar(title, xi, epoch)

        # Update best map
        fitness = results[2]  # mAP
        if fitness > best_fitness:
            best_fitness = fitness

        # Save training results
        save = (not opt.nosave) or ((not opt.evolve) and final_epoch)
        if save:
            with open('results.txt', 'r') as file:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': file.read(),
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, last)
            if opt.bucket:
                os.system('gsutil cp %s gs://%s' % (last, opt.bucket))  # upload to bucket

            # Save best checkpoint
            if best_fitness == fitness:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, wdir + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

    # Report time
    plot_results()  # save as results.png
    print('%g epochs completed in %.3f hours.' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=273)  # 500200 batches at bs 16, 117263 images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=32)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate', type=int, default=2, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='*.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--transfer', action='store_true', help='transfer learning')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--img-weights', action='store_true', help='select training images by weight')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='', help='initial weights')  # i.e. weights/darknet.53.conv.74
    parser.add_argument('--arc', type=str, default='default', help='yolo architecture')  # default, uCE, uBCE
    opt = parser.parse_args()
    opt.weights = 'weights/last.pt' if opt.resume else opt.weights
    print(opt)

    tb_writer = None
    if not opt.evolve:  # Train normally
        try:
            # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter()
        except:
            pass

        results = train()

    else:  # Evolve hyperparameters (optional)
        opt.notest = True  # only test final epoch
        opt.nosave = True  # only save final checkpoint
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(100):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Get best hyperparameters
                x = np.loadtxt('evolve.txt', ndmin=2)
                x = x[fitness(x).argmax()]  # select best fitness hyps
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = x[i + 7]

                # Mutate
                init_seeds(seed=int(time.time()))
                s = [.15, .15, .15, .15, .15, .15, .15, .15, .15, .00, .02, .20, .20, .20, .20, .20, .20, .20]  # sigmas
                for i, k in enumerate(hyp.keys()):
                    x = (np.random.randn(1) * s[i] + 1) ** 2.0  # plt.hist(x.ravel(), 300)
                    hyp[k] *= float(x)  # vary by sigmas

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale']
            limits = [(1e-4, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            results = train()

            # Write mutation results
            print_mutation(hyp, results, opt.bucket)

            # Plot results
            # plot_evolution_results(hyp)
