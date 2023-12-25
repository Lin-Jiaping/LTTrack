import torch
import torch.nn as nn
import torch.nn.functional as F

from submodule.LTM.models.motion_modules.social import SocialAttention
from submodule.LTM.models.motion_modules.uitls import make_mlp, get_global_noise
from submodule.LTM.models.motion_modules.common_modules import (
    TrajectoryEncoder,
    get_module_LTE,
    RelativeDecoder,
    get_input,
    GeneratorOutput
)


def xywh_to_cxcy(in_xy, in_dxdy):
    """
    (x, y) --> (cx, cy)
    (dw, dh) --> (log(w1/w2), log(h1/h2))
    """
    in_xy[..., 0] = in_xy[..., 0] + 0.5 * in_xy[..., 2]
    in_xy[..., 1] = in_xy[..., 1] + 0.5 * in_xy[..., 3]
    in_dxdy[1:, :, :2] = in_xy[1:, :, :2] - in_xy[:-1, :, :2]
    in_dxdy[1:, :, 2:] = torch.log((in_xy[1:, :, 2:] + 1e-6) / (in_xy[:-1, :, 2:] + 1e-6))
    in_dxdy[0, :, 2:] = 0.


class LongTermMotion(nn.Module):
    def __init__(self,
                 z_size,
                 encoder_h_dim,
                 decoder_h_dim,
                 social_feat_size,
                 embedding_dim,
                 inp_format,
                 long_term_emb_dim,
                 pred_len=1,
                 max_inp_len=30,
                 ):
        super().__init__()
        assert inp_format in ("rel", "abs", "abs_rel")

        self.inp_format = inp_format
        self.z_size = z_size
        self.embedding_dim = embedding_dim
        self.social_feat_size = social_feat_size
        self.decoder_h_dim = decoder_h_dim
        self.encoder_h_dim = encoder_h_dim
        self.long_term_emb_dim = long_term_emb_dim
        self.max_inp_len = max_inp_len
        self.pred_len = pred_len

        inp_size = 8 if inp_format == "abs_rel" else 4
        self.encoder = TrajectoryEncoder(
            inp_size=inp_size,
            hidden_size=encoder_h_dim,
            embedding_dim=embedding_dim,
            num_layers=1,
        )

        if self.social_feat_size > 0:
            self.social = SocialAttention(social_feat_size, encoder_h_dim)

        self.gs = nn.ModuleList()
        decoder = RelativeDecoder(
            pred_len=pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            num_layers=1,
            social_feat_size=self.social_feat_size,
            dropout=0.0,
            inp_format=inp_format,
        )
        setattr(self, "G_{}".format(0), decoder)
        self.gs.append(decoder)

        self.long_term_encoder = LongTermEncoder(inp_len=self.max_inp_len,
                                                 long_term_emb_dim=long_term_emb_dim)
        self.enc_h_to_dec_h = make_mlp(
            [encoder_h_dim + z_size + self.social_feat_size + long_term_emb_dim, decoder_h_dim],
            batch_norm=False,
        )

    def forward(self, in_xy, in_dxdy):
        """

        :param in_xy: in_xywh, Input tensor of shape (history length, batch size, 4)
        :param in_dxdy: in_dxdydwdh, Input tensor of shape (history length, batch size, 4).
        :return:
        """
        xywh_to_cxcy(in_xy, in_dxdy)
        in_xy_for_lte = in_xy[:-1]
        if in_xy_for_lte.shape[0] > self.max_inp_len:
            in_xy_for_lte = in_xy_for_lte[-self.max_inp_len:]

        short_inp_len = self.max_inp_len
        if in_xy.shape[0] > short_inp_len:
            in_xy = in_xy[-short_inp_len:]
            in_dxdy = in_dxdy[-short_inp_len:]

        batch_size = in_xy.size(1)
        sub_batches = [[0, batch_size]]
        encoder_inp = get_input(in_xy, in_dxdy, self.inp_format)
        enc_h = self.encoder(encoder_inp)
        self.enc_features = [enc_h]

        if self.social_feat_size > 0:
            social_feats = self.social(in_xy, in_dxdy, enc_h, sub_batches)
            self.enc_features.append(social_feats)
        else:
            social_feats = torch.zeros(batch_size, self.social_feat_size).cuda()

        if in_xy_for_lte.shape[0] < self.max_inp_len:
            in_xy_pad = in_xy_for_lte.clone().permute(2, 1, 0)
            in_xy_pad = F.pad(in_xy_pad,
                              pad=(self.max_inp_len - in_xy_for_lte.shape[0], 0, 0, 0, 0, 0),
                              mode="constant",
                              value=0)
            long_term_feats = self.long_term_encoder(in_xy_pad.permute(2, 1, 0))
        else:
            long_term_feats = self.long_term_encoder(in_xy_for_lte)
        self.enc_features.append(long_term_feats)

        enc_h = torch.cat(self.enc_features, -1)
        noise = torch.stack(
            [get_global_noise(self.z_size, sub_batches, "gaussian")]
        ).cuda()

        pred_xy, pred_dxdy = self.forward_all(
            in_xy,
            in_dxdy,
            enc_h,
            noise=noise,
            social_feats=social_feats
        )
        pred_dxdy = pred_dxdy.reshape(self.pred_len, -1, batch_size, 4)
        pred_xy = pred_xy.reshape(self.pred_len, -1, batch_size, 4)

        return GeneratorOutput(pred_dxdy, pred_xy)

    def forward_all(self, in_xy, in_dxdy, enc_h, noise, social_feats):
        """Runs all generators against the current batch.

        Args:
            in_xy: Input positions of shape (inp_len, batch_size, 4)
            in_dxdy: Input offsets of shape (inp_len, batch_size, 4)
            enc_h: Hidden state to initialize the decoders (LSTMs) with
                Shape (batch_size, enc dim).
            noise: Noise tensor (num_samples, batch_size, self.z_size).

        Returns:
            Two tensors of shape (pred_len, num_samples, num_gens, batch_size, 4)
             with predictions (positions and offsets) for every generator.
        """
        n_samples, b, z_size = noise.shape

        noise = noise.flatten(0, 1)
        enc_h = enc_h.repeat(n_samples, 1)
        in_xy = in_xy.repeat(1, n_samples, 1)
        in_dxdy = in_dxdy.repeat(1, n_samples, 1)
        enc_to_dec_inp = torch.cat([enc_h, noise], -1)
        social_feats = social_feats.repeat(n_samples, 1)

        # Shape: (1, num_samples * batch_size, dec dim)
        dec_h = self.enc_h_to_dec_h(enc_to_dec_inp).unsqueeze(0)
        state_tuple = (dec_h, torch.zeros_like(dec_h))

        pred_abs, pred_rel = self.gs[0](
            in_xy[-1], in_dxdy[-1], social_feats, state_tuple
        )
        out_xy = pred_abs.reshape(self.pred_len, n_samples, b, 4)
        out_dxdy = pred_rel.reshape(self.pred_len, n_samples, b, 4)
        return out_xy, out_dxdy


class LongTermEncoder(nn.Module):
    def __init__(self, inp_len=30, long_term_emb_dim=32):
        super().__init__()
        self.inp_len = inp_len

        self.xywh_rel_encoder, self.vel_encoder_cnn = get_module_LTE(inp_len)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(in_features=4 + 16, out_features=long_term_emb_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, in_xywh):
        """

        :param in_xywh: (inp_len, batch_size, 4), inp_len=30
        :return:
        """
        batch_size = in_xywh.shape[1]
        with torch.no_grad():
            # xywh_rel = in_xywh[..., 2:] / (in_xywh[..., 1:2] + 1e-6)
            xywh_rel = in_xywh[..., 2:] / (in_xywh[..., :2] + 1e-6)
            xywh_rel = xywh_rel.permute(2, 1, 0)  # (2, batch_size, inp_len)

            xyxy = in_xywh.clone()
            xyxy[..., 2:] += xyxy[..., :2]
            vel_xyxy = xyxy[1:] - xyxy[:-1]
            vel_xyxy = vel_xyxy.permute(2, 1, 0)

        # encode as hidden state
        xywh_rel_emb = self.xywh_rel_encoder(xywh_rel.unsqueeze(0)).reshape(batch_size, -1)
        vel_xyxy_emb = self.vel_encoder_cnn(vel_xyxy.unsqueeze(0)).reshape(batch_size, -1)
        long_term_emb = self.encoder_mlp(torch.concat((vel_xyxy_emb, xywh_rel_emb), dim=-1))
        return long_term_emb
