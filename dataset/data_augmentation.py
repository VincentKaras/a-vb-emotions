import torch
import random

"""
wav_augment audio transforms for data augmentation
"""

# Chain Runner here

class ChainRunner():
    

    def __init__(self, chain) -> None:
        self.chain = chain


    def __call__(self, data):

        src_info = {
            "channels": data.size(0),
            "length": data.size(1),
            "precision": 32,
            "rate": 16000.0,
            "bits_per_sample": 32
        }

        target_info = {
            "channels": 1,
            "length": data.size(1),
            "precision": 32,
            "rate": 16000.0,
            "bits_per_sample": 32
        }
        
        y = self.chain.apply(data, src_info, target_info)

        if torch.isnan(y).any() or torch.isinf(y).any():
            return data.clone()

        return y


def random_pitch_shift(a=-200, b=200):
    return random.randint(a, b)


def random_time_warp(f=1):
    return 1 + f * (random.random() - 0.5) / 5



