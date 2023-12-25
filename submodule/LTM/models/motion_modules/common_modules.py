from collections import namedtuple

import torch
from torch import nn

from .uitls import make_mlp


GeneratorOutput = namedtuple("generator_out", ["rel", "abs"])


def get_input(xy, dxdy, inp_format):
    if inp_format == "rel":
        inp = dxdy
    elif inp_format == "abs":
        inp = xy
    else:
        if xy.size(0) == (dxdy.size(0) + 1):
            dxdy = torch.cat([dxdy[0:1], dxdy], 0)
        inp = torch.cat([xy, dxdy], dim=2)
    return inp


class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        hidden_size=128,
        inp_size=4,
        num_layers=1,
        embedding_dim=None,
        return_hc=False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.inp_size = inp_size
        self.return_hc = return_hc

        if embedding_dim is None:
            lstm_input_size = inp_size
        else:
            lstm_input_size = embedding_dim
            self.embedding = nn.Linear(inp_size, embedding_dim)

        self.encoder = nn.GRU(
            input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_layers
        )

    def forward(self, inp, hc=None):
        """Encode a trajectory.

        Args:
            inp: Tensor with shape (sequence length, batch_size, inp_size)

        Returns:
             Tensor with shape (batch size, hidden size)
        """
        batch_size = inp.size(1)
        if self.embedding_dim is not None:
            inp = self.embedding(inp.reshape(-1, self.inp_size))
            inp = inp.reshape(-1, batch_size, self.embedding_dim)

        if hc is None:
            _, hc_t = self.encoder(inp)
        else:
            _, hc_t = self.encoder(inp, hc)
        return hc_t[-1]


class RelativeDecoder(nn.Module):
    def __init__(
        self,
        pred_len=12,
        embedding_dim=128,
        h_dim=128,
        num_layers=1,
        dropout=0.0,
        inp_format="abs_rel",
        social_feat_size=128,
    ):
        super(RelativeDecoder, self).__init__()

        self.pred_len = pred_len
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.inp_format = inp_format

        self.decoder = nn.GRU(embedding_dim, h_dim, num_layers, dropout=dropout)

        in_dim = 8 if inp_format == "abs_rel" else 4
        self.spatial_embedding = nn.Linear(in_dim, embedding_dim)
        self.hidden2pos = make_mlp(
            [h_dim + social_feat_size, h_dim // 2, 4], "leaky_relu", batch_norm=False
        )

    def forward(self, xy, dxdy, social_feats, state_tuple):
        """
        Inputs:
        - xy: Tensor of shape (batch, 4)
        - dxdy: Tensor of shape (batch, 4)
        - noise: Tensor of shape (batch, z_size)
        - social_feats: Tensor of shape (batch, social_feats)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)

        Output:
        - pred_traj: tensor of shape (pred_len, batch, 4)
        """
        preds = []

        last_obsv = torch.cat([xy, dxdy], dim=1)
        for _ in range(self.pred_len):
            to_embed = last_obsv
            if self.inp_format == "rel":
                to_embed = last_obsv[:, 4:]
            elif self.inp_format == "abs":
                to_embed = last_obsv[:, :4]
            decoder_input = self.spatial_embedding(to_embed).unsqueeze(0)
            _, state_tuple = self.decoder(decoder_input, state_tuple[0])

            # Feed last hidden state for prediction
            last_h = state_tuple[-1]
            pos_decoder_inp = torch.cat([last_h, social_feats], dim=1)
            new_dxdy = self.hidden2pos(pos_decoder_inp)

            new_xy = torch.zeros(last_obsv.shape[0], 4).cuda()
            new_xy[:, 2:] = last_obsv[:, 2:4] * torch.exp(new_dxdy[:, 2:])
            new_xy[:, :2] = last_obsv[:, :2] + new_dxdy[:, :2]
            new_xy[:, 0] -= 0.5 * new_xy[:, 2]
            new_xy[:, 1] -= 0.5 * new_xy[:, 3]

            last_obsv = torch.cat([new_xy, new_dxdy], dim=1)
            preds.append(last_obsv)

        preds = torch.stack(preds, dim=0)
        return preds[:, :, :4], preds[:, :, 4:]


# for LongTermEncoder
def get_module_LTE(inp_len):
    # xywh_rel_encoder
    layers_xywh_rel_encoder = [nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, groups=1, bias=False),
                               nn.ReLU(inplace=True), nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 5)),
                               nn.Linear(inp_len // 5, 4), nn.ReLU(inplace=True)]
    xywh_rel_encoder = nn.Sequential(*layers_xywh_rel_encoder)
    print("xywh_rel_encoder: \n", xywh_rel_encoder)

    # vel_encoder_cnn
    ve_out_dim = inp_len - 9
    layers_vel_encoder_cnn = [
        nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 5), stride=1, groups=1, bias=False), nn.BatchNorm2d(4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 5), stride=1, groups=1, bias=False), nn.BatchNorm2d(4),
        nn.ReLU(inplace=True), nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3)),
        nn.Linear(in_features=ve_out_dim // 3, out_features=4), nn.ReLU(inplace=True)]
    vel_encoder_cnn = nn.Sequential(*layers_vel_encoder_cnn)
    print("vel_encoder_cnn: \n", vel_encoder_cnn)

    return xywh_rel_encoder, vel_encoder_cnn
