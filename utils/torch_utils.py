import torch


def init_seeds(seed=0):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # https://pytorch.org/docs/stable/notes/randomness.html


def select_device(force_cpu=False, apex=False):
    # apex if mixed precision training https://github.com/NVIDIA/apex
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    if not cuda:
        print('Using CPU')
    if cuda:
        torch.backends.cudnn.benchmark = True  # set False for reproducible results
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        cuda_str = 'Using CUDA ' + ('Apex ' if apex else '')
        for i in range(0, ng):
            if i == 1:
                # torch.cuda.set_device(0)  # OPTIONAL: Set GPU ID
                cuda_str = ' ' * len(cuda_str)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (cuda_str, i, x[i].name, x[i].total_memory / c))

    print('')  # skip a line
    return device


def fuse_conv_and_bn(conv, bn):
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    with torch.no_grad():
        # init
        fusedconv = torch.nn.Conv2d(conv.in_channels,
                                    conv.out_channels,
                                    kernel_size=conv.kernel_size,
                                    stride=conv.stride,
                                    padding=conv.padding,
                                    bias=True)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0))
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(b_conv + b_bn)

        return fusedconv
