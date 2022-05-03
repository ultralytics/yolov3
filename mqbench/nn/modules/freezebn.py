import torch

class FrozenBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(FrozenBatchNorm2d, self).__init__(*args, **kwargs)
        self.training = False

    def train(self, mode=False):
        self.training = False
        for module in self.children():
            module.train(False)
        return self