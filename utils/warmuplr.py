import warnings

from torch.optim.lr_scheduler import _LRScheduler
from collections import Counter


class WarmupMultiStepLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Warmup part, sets the i-th component of lr to an early value for the first few
    epochs.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        threshold (int): The number of epochs to use the lower value
            Default: 3
        early_value (float): The value the learning rate should be set to for early epochs
            Default: 0.1
        index (int): The index of the learning rate array that should be changed. Changes all
                    values if index is set to -1.
            Default: -1

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.1      if epoch < 3
        >>> # lr = 0.05     if 3<= epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = WarmupMultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        last_epoch=-1,
        threshold=3,
        early_value=0.1,
        index=-1,
    ):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.threshold = threshold
        self.early_value = early_value
        self.index = index
        self.original_value = 0.0
        self.memory_set = False
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                DeprecationWarning,
            )
        if self.last_epoch < self.threshold:
            values = [group["lr"] for group in self.optimizer.param_groups]
            if not self.memory_set:
                if self.index == -1:
                    self.original_value = values
                else:
                    self.original_value = values[self.index]
                self.memory_set = True
            values[self.index] = self.early_value
            return values

        if self.last_epoch == self.threshold:
            values = [group["lr"] for group in self.optimizer.param_groups]
            if self.index == -1:
                values = self.original_value
            else:
                values[self.index] = self.original_value
            return values

        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            group["lr"] * self.gamma ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]
