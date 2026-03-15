import numpy as np


class SqrtLRScheduler:
    def __init__(self, lr, ref_nimg, warmup_nimg):
        """
        Parameters:
            lr (float): The learning rate.
            ref_nimg (int): The number of images where the lr decay = 1. Before this, there is no decay.
            warmup_nimg (int): The number of images to linearly increase the lr from 0 to lr.
        """
        self.lr = lr
        self.ref_nimg = ref_nimg
        self.warmup_nimg = warmup_nimg

    def get(self, nimg):
        lr = self.lr
        if self.ref_nimg > 0:
            lr /= np.sqrt(max(nimg / self.ref_nimg, 1))
        if self.warmup_nimg > 0:
            lr *= min(nimg / self.warmup_nimg, 1)
        return lr
    
class CosineLRScheduler:
    def __init__(self, lr, ref_nimg, warmup_nimg):
        self.lr = lr
        self.ref_nimg = ref_nimg
        self.warmup_nimg = warmup_nimg

    def get(self, nimg):
        lr = self.lr
        if self.ref_nimg > 0:
            lr *= 0.5 * (1 + np.cos(np.pi * nimg / self.ref_nimg))
        if self.warmup_nimg > 0:
            lr *= min(nimg / self.warmup_nimg, 1)
        return lr

class ConstantLRScheduler:
    def __init__(self, lr):
        self.lr = lr

    def get(self, nimg):
        return self.lr

class CosineWindowLRScheduler:
    def __init__(self, lr, decay_start_nimg, decay_end_nimg, warmup_nimg=0, min_lr_ratio=0.0):
        self.lr = lr
        self.decay_start_nimg = decay_start_nimg
        self.decay_end_nimg = decay_end_nimg
        self.warmup_nimg = warmup_nimg
        self.min_lr_ratio = min_lr_ratio

    def get(self, nimg):
        lr = self.lr

        if self.warmup_nimg > 0:
            lr *= min(nimg / self.warmup_nimg, 1)

        if self.decay_end_nimg <= self.decay_start_nimg:
            return lr

        if nimg <= self.decay_start_nimg:
            return lr

        if nimg >= self.decay_end_nimg:
            return lr * self.min_lr_ratio

        progress = (nimg - self.decay_start_nimg) / (self.decay_end_nimg - self.decay_start_nimg)
        cosine_ratio = 0.5 * (1 + np.cos(np.pi * progress))
        scaled_ratio = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_ratio
        return lr * scaled_ratio