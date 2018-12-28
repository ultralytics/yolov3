import torch


def check_cuda():
    return torch.cuda.is_available()


CUDA_AVAILABLE = check_cuda()


def init_seeds(seed=0):
    torch.manual_seed(seed)
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.cuda.set_device(0)  # OPTIONAL: Set your GPU if multiple available


def select_device(force_cpu=False):
    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if CUDA_AVAILABLE else 'cpu')
    return device
