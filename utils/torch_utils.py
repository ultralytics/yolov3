import torch


def init_seeds(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def select_device(force_cpu=False):
    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.device_count() > 1:
            print('WARNING Using GPU0 Only. Multi-GPU issue: https://github.com/ultralytics/yolov3/issues/21')
            torch.cuda.set_device(0)  # OPTIONAL: Set your GPU if multiple available
            # print('Using ', torch.cuda.device_count(), ' GPUs')

    print('Using ' + str(device) + '\n')
    return device
