import argparse
import time

from torch.utils.data import DataLoader

import test  # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
import torch.distributed as dist


def train(
        cfg,
        data_cfg,
        img_size=416,
        resume=False,
        epochs=270,
        batch_size=16,
        accumulate=1,
        multi_scale=False,
        freeze_backbone=False,
        num_workers=0
):
    weights = 'weights' + os.sep
    latest = weights + 'latest.pt'
    best = weights + 'best.pt'
    device = torch_utils.select_device()

    if multi_scale:
        img_size = 608  # initiate with maximum multi_scale size
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Configure run
    train_path = parse_data_cfg(data_cfg)['train']

    # Initialize model
    model = Darknet(cfg, img_size).to(device)

    # Optimizer
    lr0 = 0.001  # initial learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=.9,weight_decay = 0.0005)

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    if resume:  # Load previously saved PyTorch model
        checkpoint = torch.load(latest, map_location=device)  # load checkpoint
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']
        del checkpoint  # current, saved

    else:  # Initialize model with backbone (optional)
        if cfg.endswith('yolov3.cfg'):
            cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')
        elif cfg.endswith('yolov3-tiny.cfg'):
            cutoff = load_darknet_weights(model, weights + 'yolov3-tiny.conv.15')
        
    #initialize for distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,world_size=opt.world_size, rank=opt.rank)
        model = torch.nn.parallel.DistributedDataParallel(model)

    # Dataloader
    dataset = LoadImagesAndLabels(train_path, img_size=img_size, augment=True)
    if torch.cuda.device_count() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler=None    
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,sampler=train_sampler)        

    # Transfer learning (train only YOLO layers)
    # for i, (name, p) in enumerate(model.named_parameters()):
    #     p.requires_grad = True if (p.shape[0] == 255) else False

    # Set scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[54, 61], gamma=0.1)

    # Start training
    t0 = time.time()
    model_info(model)
    n_burnin = min(round(len(dataloader) / 5 + 1), 1000)  # burn-in batches
    for epoch in range(epochs):
        model.train()
        epoch += start_epoch

        print(('\n%8s%12s' + '%10s' * 7) % (
            'Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

        # Update scheduler (automatic)
        # scheduler.step()

        # Update scheduler (manual)
        lr = lr0 / 10 if epoch > 250 else lr0
        for x in optimizer.param_groups:
            x['lr'] = lr

        # Freeze backbone at epoch 0, unfreeze at epoch 1
        if freeze_backbone and epoch < 2:
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if (epoch == 0) else True

        rloss = defaultdict(float)
        for i, (imgs, targets, _, _) in enumerate(dataloader):
            # Unpad and collate targets
            for j, t in enumerate(targets):
                t[:, 0] = j
            targets = torch.cat([t[t[:, 5].nonzero()] for t in targets], 0).squeeze(1)

            nT = len(targets)
            if nT == 0:  # if no targets continue
                continue

            # Plot images with bounding boxes
            plot_images = False
            if plot_images:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 10))
                for ip in range(batch_size):
                    labels = xywh2xyxy(targets[targets[:, 0] == ip, 2:6]).numpy() * img_size
                    plt.subplot(4, 4, ip + 1).imshow(imgs[ip].numpy().transpose(1, 2, 0))
                    plt.plot(labels[:, [0, 2, 2, 0, 0]].T, labels[:, [1, 1, 3, 3, 1]].T, '.-')
                    plt.axis('off')

            # SGD burn-in
            if (epoch == 0) and (i <= n_burnin):
                lr = lr0 * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr

            # Run model
            pred = model(imgs.to(device))

            # Build targets
            target_list = build_targets(model, targets.to(device), pred)

            # Compute loss
            loss, loss_dict = compute_loss(pred, target_list)

            # Compute gradient
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            for key, val in loss_dict.items():
                rloss[key] = (rloss[key] * i + val) / (i + 1)

            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, len(dataloader) - 1),
                rloss['xy'], rloss['wh'], rloss['conf'],
                rloss['cls'], rloss['total'],
                nT, time.time() - t0)
            t0 = time.time()
            print(s)

            # Multi-Scale training (320 - 608 pixels) every 10 batches
            if multi_scale and (i + 1) % 10 == 0:
                dataset.img_size = random.choice(range(10, 20)) * 32
                print('multi_scale img_size = %g' % dataset.img_size)

        # Update best loss
        if rloss['total'] < best_loss:
            best_loss = rloss['total']

        # Save training results
        save = True
        if save:
            # Save latest checkpoint
            checkpoint = {'epoch': epoch,
                          'best_loss': best_loss,
                          'model': model.module.state_dict() if type(model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, latest)

            # Save best checkpoint
            if best_loss == rloss['total']:
                os.system('cp ' + latest + ' ' + best)

            # Save backup weights every 5 epochs (optional)
            if (epoch > 0) and (epoch % 5 == 0):
                os.system('cp ' + latest + ' ' + weights + 'backup{}.pt'.format(epoch))

        # Calculate mAP
        if type(model) is nn.parallel.DistributedDataParallel:
            model = model.module
        with torch.no_grad():
            P, R, mAP = test.test(cfg, data_cfg, weights=latest, batch_size=batch_size, img_size=img_size, model=model)

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 3 % (P, R, mAP) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=270, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='cfg/coco.data', help='coco.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--num-workers', type=int, default=4, help='number of Pytorch DataLoader workers')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,help='url used to set up distributed training')
    parser.add_argument('--rank', default=-1, type=int,help='node rank for distributed training')
    parser.add_argument('--world-size', default=-1, type=int,help='number of nodes for distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,help='distributed backend')
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
        accumulate=opt.accumulate,
        multi_scale=opt.multi_scale,
        num_workers=opt.num_workers
    )
