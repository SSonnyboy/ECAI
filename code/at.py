import torch
import numpy as np
import random
import torch.nn.functional as F


def at(x):
    y = F.normalize(x.pow(2).mean(1), dim=1)
    y = y.view(y.size(0), -1)
    return y


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()


if __name__ == "__main__":
    x1 = torch.rand((4, 64, 128, 128, 20))
    x2 = torch.rand((4, 64, 128, 128, 20))
    print(at_loss(x1, x2))
