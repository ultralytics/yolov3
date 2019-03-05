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
        device = torch.device('cuda' if cuda else 'cpu')
        num_devices = torch.cuda.device_count()
        device_info = [torch.cuda.get_device_properties(n) for n in range(num_devices)]
        print("Using CUDA. Available devices: ")
        for i in range(num_devices):
            print(f"{i} - {device_info[i].name} - {device_info[i].total_memory//1024**2}MB")
    return device
