import argparse
import time

import test  # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *


def train(
        cfg,
        data_cfg,
        img_size=416,
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        multi_scale=False,
        freeze_backbone=False,
        var=0,
):
    weights = 'weights' + os.sep
    latest = weights + 'latest.pt'
    best = weights + 'best.pt'
    device = torch_utils.select_device()

    if multi_scale:  # pass maximum multi_scale size
        img_size = 608
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Configure run
    train_path = parse_data_cfg(data_cfg)['train']

    # Initialize model
    model = Darknet(cfg, img_size)

    # Get dataloader
    dataloader = LoadImagesAndLabels(train_path, batch_size, img_size, multi_scale=multi_scale, augment=True)

    lr0 = 0.001
    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    if resume:
        checkpoint = torch.load(latest, map_location='cpu')

        # Load weights to resume from
        model.load_state_dict(checkpoint['model'])

        # if torch.cuda.device_count() > 1:
        #   model = nn.DataParallel(model)
        model.to(device).train()

        # Transfer learning (train only YOLO layers)
        # for i, (name, p) in enumerate(model.named_parameters()):
        #     p.requires_grad = True if (p.shape[0] == 255) else False

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']

        del checkpoint  # current, saved

    else:
        # Initialize model with backbone (optional)
        if cfg.endswith('yolov3.cfg'):
            load_darknet_weights(model, weights + 'darknet53.conv.74')
            cutoff = 75
        elif cfg.endswith('yolov3-tiny.cfg'):
            load_darknet_weights(model, weights + 'yolov3-tiny.conv.15')
            cutoff = 15

        # if torch.cuda.device_count() > 1:
        #    model = nn.DataParallel(model)
        model.to(device).train()

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)

    # Set scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[54, 61], gamma=0.1)

    # Start training
    t0 = time.time()
    model_info(model)
    n_burnin = min(round(dataloader.nB / 5 + 1), 1000)  # number of burn-in batches
    for epoch in range(epochs):
        epoch += start_epoch

        print(('%8s%12s' + '%10s' * 7) % (
            'Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

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
        if freeze_backbone and (epoch < 2):
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if (epoch == 0) else True

        ui = -1
        rloss = defaultdict(float)  # running loss
        optimizer.zero_grad()
        for i, (imgs, targets, _, _) in enumerate(dataloader):
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue

            # SGD burn-in
            if (epoch == 0) & (i <= n_burnin):
                lr = lr0 * (i / n_burnin) ** 4
                for g in optimizer.param_groups:
                    g['lr'] = lr

            # Compute loss
            loss = model(imgs.to(device), targets, var=var)

            # Compute gradient
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            ui += 1
            for key, val in model.losses.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)

            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, len(dataloader) - 1),
                rloss['xy'], rloss['wh'], rloss['conf'],
                rloss['cls'], rloss['loss'],
                model.losses['nT'], time.time() - t0)
            t0 = time.time()
            print(s)

        # Update best loss
        if rloss['loss'] < best_loss:
            best_loss = rloss['loss']

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, latest)

        # Save best checkpoint
        if best_loss == rloss['loss']:
            os.system('cp ' + latest + ' ' + best)

        # Save backup weights every 5 epochs (optional)
        # if (epoch > 0) & (epoch % 5 == 0):
        #     os.system('cp ' + latest + ' ' + weights + 'backup{}.pt'.format(epoch)))

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
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='cfg/coco.data', help='coco.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
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
        multi_scale=opt.multi_scale,
        var=opt.var,
    )
