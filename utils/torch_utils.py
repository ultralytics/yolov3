import torch


def init_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(force_cpu=False):
    if force_cpu:
        cuda = False
        device = torch.device('cpu')
    else:
        cuda = torch.cuda.is_available()
        device = torch.device('cuda:0' if cuda else 'cpu')

        if torch.cuda.device_count() > 1:
            print('Found %g GPUs' % torch.cuda.device_count())
            print('WARNING Multi-GPU Issue: https://github.com/ultralytics/yolov3/issues/21')
            torch.cuda.set_device(0)  # OPTIONAL: Set your GPU if multiple available
        #    # print('Using ', torch.cuda.device_count(), ' GPUs')

    print('Using %s %s\n' % (device.type, torch.cuda.get_device_properties(0) if cuda else ''))
    return device
