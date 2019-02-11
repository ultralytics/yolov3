import argparse
import time

from models import *
from utils.datasets import *
from utils.utils import *

from utils import torch_utils

# Import test.py to get mAP after each epoch
import test


def train(
        cfg,
        data_cfg,
        img_size=416,
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        weights='weights',
        multi_scale=False,
        freeze_backbone=True,
        var=0,
):
    device = torch_utils.select_device()

    if multi_scale:  # pass maximum multi_scale size
        img_size = 608
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    latest = os.path.join(weights, 'latest.pt')
    best = os.path.join(weights, 'best.pt')

    # Configure run
    data_cfg = parse_data_cfg(data_cfg)
    train_path = data_cfg['train']

    # Initialize model
    model = Darknet(cfg, img_size)

    # Get dataloader
    dataloader = LoadImagesAndLabels(train_path, batch_size, img_size, multi_scale=multi_scale, augment=True)

    lr0 = 0.001
    if resume:
        checkpoint = torch.load(latest, map_location='cpu')

        model.load_state_dict(checkpoint['model'])
        if torch.cuda.device_count() > 1:
            raise Exception('Multi-GPU issue: https://github.com/ultralytics/yolov3/issues/21')
            # print('Using ', torch.cuda.device_count(), ' GPUs')
            # model = nn.DataParallel(model)
        model.to(device).train()

        # # Transfer learning (train only YOLO layers)
        # for i, (name, p) in enumerate(model.named_parameters()):
        #     if p.shape[0] != 650:  # not YOLO layer
        #         p.requires_grad = False

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']

        del checkpoint  # current, saved

    else:
        start_epoch = 0
        best_loss = float('inf')

        # Initialize model with darknet53 weights (optional)
        load_darknet_weights(model, os.path.join(weights, 'darknet53.conv.74'))

        if torch.cuda.device_count() > 1:
            raise Exception('Multi-GPU not currently supported: https://github.com/ultralytics/yolov3/issues/21')
            # print('Using ', torch.cuda.device_count(), ' GPUs')
            # model = nn.DataParallel(model)
        model.to(device).train()

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)

    # Set scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[54, 61], gamma=0.1)

    model_info(model)
    t0 = time.time()
    for epoch in range(epochs):
        epoch += start_epoch

        print(('%8s%12s' + '%10s' * 9) % (
            'Epoch', 'Batch', 'x', 'y', 'w', 'h', 'conf', 'cls', 'total', 'nTargets', 'time'))

        # Update scheduler (automatic)
        # scheduler.step()

        # Update scheduler (manual)  at 0, 54, 61 epochs to 1e-3, 1e-4, 1e-5
        if epoch > 50:
            lr = lr0 / 10
        else:
            lr = lr0
        for g in optimizer.param_groups:
            g['lr'] = lr

        # Freeze darknet53.conv.74 for first epoch
        if freeze_backbone:
            if epoch == 0:
                for i, (name, p) in enumerate(model.named_parameters()):
                    if int(name.split('.')[1]) < 75:  # if layer < 75
                        p.requires_grad = False
            elif epoch == 1:
                for i, (name, p) in enumerate(model.named_parameters()):
                    if int(name.split('.')[1]) < 75:  # if layer < 75
                        p.requires_grad = True

        ui = -1
        rloss = defaultdict(float)  # running loss
        optimizer.zero_grad()
        for i, (imgs, targets) in enumerate(dataloader):
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue

            # SGD burn-in
            if (epoch == 0) & (i <= 1000):
                lr = lr0 * (i / 1000) ** 4
                for g in optimizer.param_groups:
                    g['lr'] = lr

            # Compute loss, compute gradient, update parameters
            loss = model(imgs.to(device), targets, var=var)
            loss.backward()

            # accumulate gradient for x batches before optimizing
            if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            ui += 1
            for key, val in model.losses.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)

            s = ('%8s%12s' + '%10.3g' * 9) % (
                '%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, len(dataloader) - 1), rloss['x'],
                rloss['y'], rloss['w'], rloss['h'], rloss['conf'], rloss['cls'],
                rloss['loss'], model.losses['nT'], time.time() - t0)
            t0 = time.time()
            print(s)

        # Update best loss
        loss_per_target = rloss['loss'] / rloss['nT']
        if loss_per_target < best_loss:
            best_loss = loss_per_target

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, latest)

        # Save best checkpoint
        if best_loss == loss_per_target:
            os.system('cp ' + latest + ' ' + best)

        # Save backup weights every 5 epochs
        if (epoch > 0) & (epoch % 5 == 0):
            os.system('cp ' + latest + ' ' + os.path.join(weights, 'backup{}.pt'.format(epoch)))

        # Calculate mAP
        with torch.no_grad():
            mAP, R, P = test.test(cfg, data_cfg, weights=latest, batch_size=batch_size, img_size=img_size)

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 3 % (mAP, P, R) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
    parser.add_argument('--data-cfg', type=str, default='cfg/coco.data', help='path to data config file')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--weights', type=str, default='weights', help='path to store weights')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--freeze', action='store_true', help='freeze darknet53.conv.74 layers for first epoch')
    parser.add_argument('--var', type=float, default=0, help='test variable')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()

    train(
        opt.cfg,
        opt.data_cfg,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        weights=opt.weights,
        multi_scale=opt.multi_scale,
        freeze_backbone=opt.freeze,
        var=opt.var,
    )
