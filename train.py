import argparse
import time
import os

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import test
from models import *
from utils.datasets import *
from utils.utils import *

# Training hyperparameters
# hyp = {'giou': 0.8541,  # giou loss gain
#        'xy': 4.062,  # xy loss gain
#        'wh': 0.1845,  # wh loss gain
#        'cls': 21.61,  # cls loss gain
#        'cls_pw': 1.957,  # cls BCELoss positive_weight
#        'obj': 22.9,  # obj loss gain
#        'obj_pw': 2.894,  # obj BCELoss positive_weight
#        'iou_t': 0.3689,  # iou target-anchor training threshold
#        'lr0': 0.001844,  # initial learning rate
#        'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
#        'momentum': 0.913,  # SGD momentum
#        'weight_decay': 0.000467,  # optimizer weight decay
#        'hsv_s': 0.8,  # image HSV-Saturation augmentation (fraction)
#        'hsv_v': 0.388,  # image HSV-Value augmentation (fraction)
#        'degrees': 1.2,  # image rotation (+/- deg)
#        'translate': 0.119,  # image translation (+/- fraction)
#        'scale': 0.0589,  # image scale (+/- gain)
#        'shear': 0.401}  # image shear (+/- deg)

# Evolved hyp
hyp = {'giou': 1.582,  # giou loss gain
       'xy': 4.688,  # xy loss gain
       'wh': 0.1857,  # wh loss gain
       'cls': 27.76,  # cls loss gain
       'cls_pw': 1.446,  # cls BCELoss positive_weight
       'obj': 21.35,  # obj loss gain
       'obj_pw': 3.941,  # obj BCELoss positive_weight
       'iou_t': 0.2635,  # iou target-anchor training threshold
       'lr0': 0.002324,  # initial learning rate
       'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.97,  # SGD momentum
       'weight_decay': 0.0004569 ,  # optimizer weight decay
       'hsv_s': 0.5703,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.3174,  # image HSV-Value augmentation (fraction)
       'degrees': 1.113,  # image rotation (+/- deg)
       'translate': 0.06797,  # image translation (+/- fraction)
       'scale': 0.1059,  # image scale (+/- gain)
       'shear': 0.5768}  # image shear (+/- deg)


#       0.207      0.496      0.353      0.283       3.09   ||   1.582      4.688     0.1857      27.76      1.446      21.35      3.941     0.2635   0.002324         -4       0.97  0.0004569     0.5703     0.3174      1.113    0.06797     0.1059     0.5768


def train(cfg,
          data_cfg,
          img_size=416,
          epochs=100,  # 500200 batches at bs 16, 117263 images = 273 epochs
          batch_size=16,
          accumulate=4):  # effective bs = batch_size * accumulate = 8 * 8 = 64
    # Initialize
    os.makedirs(opt.output_dir, exist_ok=True)
    init_seeds()
    weights = 'weights' + os.sep
    last = os.path.join(opt.output_dir, 'last.pt')
    best = os.path.join(opt.output_dir, 'best.pt')
    device = torch_utils.select_device()
    multi_scale = opt.multi_scale

    if multi_scale:
        img_size_min = round(img_size / 32 / 1.5)
        img_size_max = round(img_size / 32 * 1.5)
        img_size = img_size_max * 32  # initiate with maximum multi_scale size

    # Configure run
    data_dict = parse_data_cfg(data_cfg)
    train_path = data_dict['train']
    nc = int(data_dict['classes'])  # number of classes

    # Initialize model
    model = Darknet(cfg, img_size).to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_fitness = 0.0
    best_results = None
    best_efit = 0.0
    if opt.resume: # resume from ckpt
        chkpt = torch.load(opt.resume, map_location=device)  # load checkpoint
        model.load_state_dict(chkpt['model'])

        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        if chkpt.get('training_results') is not None:
            with open('results.txt', 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt

    else:  # Initialize model with backbone (optional)
        if opt.pretrained_weights:
            cutoff = load_darknet_weights(model, opt.pretrained_weights)

    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    # lf = lambda x: 1 - x / epochs  # linear ramp to zero
    # lf = lambda x: 10 ** (hyp['lrf'] * x / epochs)  # exp ramp
    # lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inverse exp ramp
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in (0.8, 0.9)], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    # Dataset
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  augment=True,
                                  hyp=hyp,
                                  rect=opt.rect)  # rectangular training
    
    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank

        model = torch.nn.parallel.DistributedDataParallel(model)
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    
    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=opt.num_workers,
                            shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    # Mixed precision training https://github.com/NVIDIA/apex
    mixed_precision = True
    if mixed_precision:
        try:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
        except:  # not installed: install help: https://github.com/NVIDIA/apex/issues/259
            mixed_precision = False

    # Start training
    model.hyp = hyp  # attach hyperparameters to model
    model_info(model, report='summary')  # 'full' or 'summary'
    nb = len(dataloader)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    n_burnin = min(round(nb / 5 + 1), 1000)  # burn-in batches
    t, t0 = time.time(), time.time()
    torch.cuda.empty_cache()
    tr_losses = []
    val_results = []
    for epoch in range(start_epoch, epochs):
        model.train()
        print(('\n%8s%12s' + '%10s' * 7) %
              ('Epoch', 'Batch', 'GIoU/xy', 'wh', 'obj', 'cls', 'total', 'targets', 'img_size'))

        # Update scheduler
        scheduler.step()

        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional), currently not freezing
        freeze_backbone = False
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75, cutoff is currently -1 however
                    p.requires_grad = False if epoch == 0 else True

        mloss = torch.zeros(5).to(device)  # mean losses
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Multi-Scale training TODO: short-side to 32-multiple https://github.com/ultralytics/yolov3/issues/358
            if multi_scale:
                if (i + nb * epoch) / accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                    img_size = random.choice(range(img_size_min, img_size_max + 1)) * 32
                    # print('img_size = %g' % img_size)
                scale_factor = img_size / max(imgs.shape[-2:])
                imgs = F.interpolate(imgs, scale_factor=scale_factor, mode='bilinear', align_corners=False)

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                lr = hyp['lr0'] * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr

            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model, giou_loss=not opt.xywh)
            
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
            if (i + 1) % accumulate == 0 or (i + 1) == nb:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, nb - 1), *mloss, len(targets), img_size)
            t = time.time()
            pbar.set_description(s)  # print(s)

        # Calculate mAP (always test final epoch)
        if not (opt.notest or opt.nosave) or epoch == epochs - 1:
            with torch.no_grad():
                results, maps = test.test(cfg, data_cfg, opt.output_dir, opt.data_dir, batch_size=batch_size, img_size=opt.img_size, model=model, conf_thres=0.1, save_json=True)
                val_results.append([*results, maps])
            tr_losses.append(mloss)

        # Write epoch results
        with open(os.path.join(opt.output_dir, 'results.txt'), 'a') as file:
            file.write(s + '%11.3g' * 5 % results + '\n')  # P, R, mAP, F1, test_loss

        # Update best map
        fitness = results[2]
        efit = results[2] * 0.5 + results[3] * 0.5
        if fitness > best_fitness:
            best_fitness = fitness
        if efit > best_efit:     
            best_results = results
            best_efit = efit

        # Save training results
        save = (not opt.nosave) or (epoch == epochs - 1)
        if save:
            with open(os.path.join(opt.output_dir, 'results.txt'), 'r') as file:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': file.read(),
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': optimizer.state_dict()}

            # Save last checkpoint (replaced every epoch)
            torch.save(chkpt, last)

            # Save best checkpoint (replaced every epoch if best)
            if best_fitness == fitness:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 9:
                torch.save(chkpt, os.path.join(opt.output_dir, 'backup%g.pt' % epoch))

            # Delete checkpoint
            del chkpt

            # save losses for tr and val in files, will be replaced every epoch since experiment might be ended prematurely
            torch.save(tr_losses, os.path.join(opt.output_dir, 'tr_losses'))
            torch.save(val_results, os.path.join(opt.output_dir, 'val_results'))
            
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None 
    torch.cuda.empty_cache() 
    
    return best_results


def print_mutation(hyp, results):
    # Write mutation results
    a = '%11s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%11.4g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%11.3g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))
    
    with open(os.path.join(opt.output_dir, 'evolve.txt'), 'a') as f:
        f.write(c + b + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--accumulate', type=int, default=16, help='number of batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='train at (1/1.5)x - 1.5x sizes')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', type=str, default='', help='resume training from this checkpoint')
    parser.add_argument('--num-workers', type=int, default=4, help='number of Pytorch DataLoader workers')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--xywh', action='store_true', help='use xywh loss instead of GIoU loss')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--output-dir', type=str, help='where the checkpoints, results are stored')
    parser.add_argument('--data-dir', type=str, default='/data', help='where the data is stored')
    parser.add_argument('--pretrained-weights', type=str, default='', help='path to weights')
    parser.add_argument('--gpu', type=str, default="0", help='which gpu(s) to use on a server')
    opt = parser.parse_args()
    print(opt)
    
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
    
    os.makedirs(opt.output_dir, exist_ok=True)

    # Train
    orig_dir = opt.output_dir
    results = train(opt.cfg,
                    opt.data_cfg,
                    img_size=opt.img_size,
                    epochs=opt.epochs,
                    batch_size=opt.batch_size,
                    accumulate=opt.accumulate)
    
    # Evolve hyperparameters (optional)
    if opt.evolve:
        gen = 1000  # generations to evolve
        print_mutation(hyp, results)  # Write mutation results
        for n in range(gen):
            # Get best hyperparameters
            x = np.loadtxt(os.path.join(opt.output_dir, 'evolve.txt'), ndmin=2)
            opt.output_dir = os.path.join(orig_dir, 'gen' + str(n)) 
            fitness = x[:, 2] * 0.5 + x[:, 3] * 0.5  # fitness as weighted combination of mAP and F1
            x = x[fitness.argmax()]  # select best fitness hyps
            for i, k in enumerate(hyp.keys()):
                hyp[k] = x[i + 5]

            # Mutate
            init_seeds(seed=int(time.time()))
            s = [.15, .15, .15, .15, .15, .15, .15, .15, .15, .00, .05, .20, .20, .20, .20, .20, .20, .20]
            for i, k in enumerate(hyp.keys()):
                x = (np.random.randn(1) * s[i] + 1) ** 2.0  # plt.hist(x.ravel(), 300)
                hyp[k] *= float(x)  # vary by 20% 1sigma

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale']
            limits = [(1e-4, 1e-2), (0.00, 0.70), (0.60, 0.97), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            results = train(opt.cfg,
                            opt.data_cfg,
                            img_size=opt.img_size,
                            epochs=opt.epochs,
                            batch_size=opt.batch_size,
                            accumulate=opt.accumulate)

            # Write mutation results
            opt.output_dir = orig_dir
            print_mutation(hyp, results)
