import torch
from torch import nn


def make_mlp(dim_list, activation="relu", batch_norm=False, dropout=0):
    layers = []
    if len(dim_list) > 2:
        for dim_in, dim_out in zip(dim_list[:-2], dim_list[1:-1]):
            layers.append(nn.Linear(dim_in, dim_out))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(dim_list[-2], dim_list[-1]))
    model = nn.Sequential(*layers)
    return model


def gan_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape)
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0)
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


def get_global_noise(dim, sub_batches, noise_type):
    noise = []
    for start, end in sub_batches:
        n = gan_noise((1, dim), noise_type)
        noise.append(n.repeat(end - start, 1))
    return torch.cat(noise)
