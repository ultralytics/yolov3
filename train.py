import argparse
import time

from models import *
from utils.datasets import *
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=160, help='number of epochs')
parser.add_argument('-batch_size', type=int, default=12, help='size of each image batch')
parser.add_argument('-data_config_path', type=str, default='cfg/coco.data', help='data config file path')
parser.add_argument('-cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
parser.add_argument('-img_size', type=int, default=32 * 13, help='size of each image dimension')
parser.add_argument('-resume', default=False, help='resume training flag')
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if cuda:
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True


def main(opt):
    os.makedirs('checkpoints', exist_ok=True)

    # Configure run
    data_config = parse_data_config(opt.data_config_path)
    num_classes = int(data_config['classes'])
    if platform == 'darwin':  # MacOS (local)
        train_path = data_config['train']
    else:  # linux (cloud, i.e. gcp)
        train_path = '../coco/trainvalno5k.part'

    # Initialize model
    model = Darknet(opt.cfg, opt.img_size)

    # Get dataloader
    dataloader = load_images_and_labels(train_path, batch_size=opt.batch_size, img_size=opt.img_size, augment=True)

    # Reload saved optimizer state
    start_epoch = 0
    best_loss = float('inf')
    if opt.resume:
        checkpoint = torch.load('checkpoints/latest.pt', map_location='cpu')

        model.load_state_dict(checkpoint['model'])
        if torch.cuda.device_count() > 1:
            print('Using ', torch.cuda.device_count(), ' GPUs')
            model = nn.DataParallel(model)
        model.to(device).train()

        # # Transfer learning
        # for i, (name, p) in enumerate(model.named_parameters()):
        #     #name = name.replace('module_list.', '')
        #     #print('%4g %70s %9s %12g %20s %12g %12g' % (
        #     #    i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
        #     if p.shape[0] != 650:  # not YOLO layer
        #         p.requires_grad = False

        # Set optimizer
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3,
                                    momentum=.9, weight_decay=5e-4, nesterov=True)

        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']

        del checkpoint  # current, saved
    else:
        if torch.cuda.device_count() > 1:
            print('Using ', torch.cuda.device_count(), ' GPUs')
            model = nn.DataParallel(model)
        model.to(device).train()

        # Set optimizer
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=.9, weight_decay=5e-4, nesterov=True)

    # Set scheduler
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99082, last_epoch=start_epoch - 1)

    modelinfo(model)
    t0, t1 = time.time(), time.time()
    print('%10s' * 16 % (
        'Epoch', 'Batch', 'x', 'y', 'w', 'h', 'conf', 'cls', 'total', 'P', 'R', 'nTargets', 'TP', 'FP', 'FN', 'time'))
    for epoch in range(opt.epochs):
        epoch += start_epoch

        # Multi-Scale YOLO Training
        # img_size = random.choice(range(10, 20)) * 32  # 320 - 608 pixels
        # dataloader = load_images_and_labels(train_path, batch_size=opt.batch_size, img_size=img_size, augment=True)
        # print('Running this epoch with image size %g' % img_size)

        # Update scheduler (automatic)
        # scheduler.step()

        # Update scheduler (manual)
        # for g in optimizer.param_groups:
        #     g['lr'] = 1e-3 * (g ** epoch)  # 1/10th every [30, 50, 100, 250] epochs using g = [.926, .955, .977, .992]

        ui = -1
        rloss = defaultdict(float)  # running loss
        metrics = torch.zeros(4, num_classes)
        for i, (imgs, targets) in enumerate(dataloader):
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue

            # SGD burn-in
            if (epoch == 0) & (i <= 1000):
                power = 4
                lr = 1e-3 * (i / 1000) ** power
                for g in optimizer.param_groups:
                    g['lr'] = lr
                # print('SGD Burn-In LR = %9.5g' % lr, end='')

            # Compute loss, compute gradient, update parameters
            loss = model(imgs.to(device), targets, requestPrecision=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute running epoch-means of tracked metrics
            ui += 1
            metrics += model.losses['metrics']
            for key, val in model.losses.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)

            # Precision
            precision = metrics[0] / (metrics[0] + metrics[1] + 1e-16)
            k = (metrics[0] + metrics[1]) > 0
            if k.sum() > 0:
                mean_precision = precision[k].mean()
            else:
                mean_precision = 0

            # Recall
            recall = metrics[0] / (metrics[0] + metrics[2] + 1e-16)
            k = (metrics[0] + metrics[2]) > 0
            if k.sum() > 0:
                mean_recall = recall[k].mean()
            else:
                mean_recall = 0

            s = ('%10s%10s' + '%10.3g' * 14) % (
                '%g/%g' % (epoch, opt.epochs - 1), '%g/%g' % (i, len(dataloader) - 1), rloss['x'],
                rloss['y'], rloss['w'], rloss['h'], rloss['conf'], rloss['cls'],
                rloss['loss'], mean_precision, mean_recall, model.losses['nT'], model.losses['TP'],
                model.losses['FP'], model.losses['FN'], time.time() - t1)
            t1 = time.time()
            print(s)

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '\n')

        # Update best loss
        loss_per_target = rloss['loss'] / rloss['nT']
        if loss_per_target < best_loss:
            best_loss = loss_per_target

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, 'checkpoints/latest.pt')

        # Save best checkpoint
        if best_loss == loss_per_target:
            os.system('cp checkpoints/latest.pt checkpoints/best.pt')

        # Save backup checkpoint
        if (epoch > 0) & (epoch % 5 == 0):
            os.system('cp checkpoints/latest.pt checkpoints/backup' + str(epoch) + '.pt')

    # Save final model
    dt = time.time() - t0
    print('Finished %g epochs in %.2fs (%.2fs/epoch)' % (epoch, dt, dt / (epoch + 1)))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main(opt)
    torch.cuda.empty_cache()
