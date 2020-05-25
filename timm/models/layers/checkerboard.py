import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class CheckerBoard2D(nn.Module):
    def __init__(self, drop_prob, drop_size, drop_shape):
        super(CheckerBoard2D, self).__init__()

        self.drop_prob = drop_prob
        self.drop_size = drop_size
        self.drop_shape = drop_shape

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            mask = torch.ones_like(x).float().to(x.device)
            prob = np.random.rand(x.shape[1])
            indexes = np.where(prob <= self.drop_prob)
            _, _, height, width = x.shape

            if self.drop_shape == 'square':
                shape = round(height / 2 * self.drop_size)
                for i in indexes:
                    mask[:, i, :, :] = torch.from_numpy(
                        np.kron([[1, 0] * shape, [0, 1] * shape] * shape,
                                np.ones((shape, shape))))[:height, :width].float().to(x.device)

            if self.drop_shape == 'rectangle':
                shape = round(height / 2 * self.drop_size)
                ver, hor = np.random.random_sample(2)
                for i in indexes:
                    if ver > hor:
                        mask[:, i, :, :] = torch.from_numpy(
                            np.kron([[1, 0] * int((shape / 2)), [0, 1] * int((shape / 2))] * 2 * shape,
                                    np.ones((shape, 2 * shape))))[:height, :width].float().to(x.device)
                    else:
                        mask[:, i, :, :] = torch.from_numpy(
                            np.kron([[1, 0] * 2 * shape, [0, 1] * 2 * shape] * int((shape / 2)),
                                    np.ones((2 * shape, shape))))[:height, :width].float().to(x.device)

            x = x * mask

            return x


class CheckerBoardScheduler(nn.Module):
    def __init__(self, checkerboard, start_value, stop_value, nr_steps):
        super(CheckerBoardScheduler, self).__init__()
        self.checkerboard = checkerboard
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=int(nr_steps))

    def forward(self, x):
        out = self.checkerboard(x)
        if self.i > len(self.drop_values):
            self.checkerboard.drop_prob = self.drop_values[self.i]

        self.i += 1
        return out
