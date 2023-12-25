import torch
from torch import nn


class AssaPosCost(nn.Module):
    def __init__(self,
                 rel_pos_emb_dim=16,
                 trk_det_encoder_emb_dim=8,
                 rel_k=5,
                 mode='train'):
        super().__init__()
        trk_encoder_hsize = 0
        self.rel_pos_encoder = RelPosEncoder(emb_dim=rel_pos_emb_dim, k=rel_k)
        self.trk_det_encoder = TrkDetEncoder(emb_dim=trk_det_encoder_emb_dim)
        decoder_layers = [nn.Linear(trk_encoder_hsize + rel_pos_emb_dim + trk_det_encoder_emb_dim, 8),
                          nn.LeakyReLU(inplace=True),
                          nn.Linear(8, 1)]
        if mode != 'train':
            decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, trk_seq, trk_rel_pos, det_bbox, det_rel_pos):
        """

        :param trk_seq: [B, M, seq_len, 4], xyxy
        :param trk_rel_pos: [B, M, 5, 4]
        :param det_bbox: [B, N, 4]
        :param det_rel_pos: [B, N, 5, 4]
        :return:
        """
        rel_pos_emb = self.rel_pos_encoder(trk_rel_pos, det_rel_pos, trk_seq[:, :, -1], det_bbox)
        trk_det_emb = self.trk_det_encoder(trk_seq[:, :, -1], det_bbox)
        scene_emb = torch.concat((rel_pos_emb, trk_det_emb), dim=-1)
        simi_matrix = self.decoder(scene_emb)
        return simi_matrix


class TrkEncoder(nn.Module):
    def __init__(
            self,
            hidden_size=32,
            inp_size=4,
            num_layers=1,
            embedding_dim=16,
            return_hc=False
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

    def forward(self, trk_seq, det_bbox, hc=None):
        """Encode a trajectory.

        Args:
            trk_seq: [B, M, seq_len, 4]
            det_bbox: [B, N, 4]
            inp: Tensor with shape (sequence length, batch_size, inp_size)
            hc: hidden state, [B, M, N, hsize] ?

        Returns:
             Tensor with shape (batch size, hidden size)
        """
        trk_num = trk_seq.shape[1]
        det_num = det_bbox.shape[1]
        trk_seq = torch.unsqueeze(trk_seq, dim=2).repeat(1, 1, det_num, 1, 1)  # [B, M, N, seq_len, 4]
        det_bbox = torch.unsqueeze(det_bbox.unsqueeze(2), dim=1).repeat(1, trk_num, 1, 1, 1)  # [B, M, N, 1, 4]
        trk_det_seq = torch.concat((trk_seq, det_bbox), dim=3)
        inp = torch.flatten(trk_det_seq, start_dim=0, end_dim=2).permute(1, 0, 2)

        batch_size = inp.size(1)
        if self.embedding_dim is not None:
            inp = self.embedding(inp.reshape(-1, self.inp_size))
            inp = inp.reshape(-1, batch_size, self.embedding_dim)

        if hc is None:
            _, hc_t = self.encoder(inp)
        else:
            hc = hc.reshape(hc.shape[0], -1, hc.shape[-1])
            _, hc_t = self.encoder(inp, hc)

        h_t = hc_t.reshape(hc_t.shape[0], -1, trk_num, det_num, hc_t.shape[-1])  # [seq_len, B, M, N, hsize]
        return h_t[-1]


class RelPosEncoder(nn.Module):
    def __init__(self, emb_dim=16, k=5):
        super().__init__()
        self.k = k
        input_dim = k * 4 + 4

        self.fc_1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True)
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, emb_dim)
        )

    def forward(self, trk_rel_pos, det_rel_pos, trk_bbox=None, det_bbox=None):
        """

        :param trk_bbox: [B, M, 4]
        :param det_bbox: [B, N, 4]
        :param trk_rel_pos: [B, M, 5, 4]
        :param det_rel_pos: [B, N, 5, 4]
        :return:
        """
        trk_rel_pos = trk_rel_pos[:, :, :self.k, :]
        det_rel_pos = det_rel_pos[:, :, :self.k, :]
        trk_num = trk_rel_pos.shape[1]
        det_num = det_rel_pos.shape[1]
        rel_pos = torch.concat((trk_rel_pos, det_rel_pos), dim=1)
        rel_pos = torch.flatten(rel_pos, start_dim=2, end_dim=3)  # [B, M+N, 20]

        bboxes = torch.concat((trk_bbox, det_bbox), dim=1)
        rel_pos = torch.concat((bboxes, rel_pos), dim=-1)

        rel_pos_emb = self.fc_1(rel_pos)
        trk_emb = torch.unsqueeze(rel_pos_emb[:, :trk_num], dim=2).repeat(1, 1, det_num, 1)  # [B, M, N, 128]
        det_emb = torch.unsqueeze(rel_pos_emb[:, trk_num:], dim=1).repeat(1, trk_num, 1, 1)
        rel_pos_emb = torch.concat((trk_emb, det_emb), dim=-1)
        rel_pos_emb = self.fc_2(rel_pos_emb)
        return rel_pos_emb


class TrkDetEncoder(nn.Module):
    def __init__(self, input_dim=4, emb_dim=8):
        super().__init__()
        mlp_layers = [nn.Linear(input_dim, emb_dim),
                      nn.LeakyReLU(inplace=True)]
        self.encoder = nn.Sequential(*mlp_layers)

    def forward(self, trk_bbox, det_bbox):
        """

        :param trk_bbox: Tensor(B, M, xyxy)
        :param det_bbox: Tensor(B, N, xyxy)
        :return: trkdet_emb: Tensor(1, M, N, emb_dim)
        """
        with torch.no_grad():
            trk_bbox = torch.unsqueeze(trk_bbox, dim=2)
            det_bbox = torch.unsqueeze(det_bbox, dim=1)
            trk_bbox = trk_bbox.repeat(1, 1, det_bbox.shape[2], 1)
            det_bbox = det_bbox.repeat(1, trk_bbox.shape[1], 1, 1)

            center_dis = ((trk_bbox[..., 2:] - trk_bbox[..., :2]) / 2 + trk_bbox[..., :2]) - (
                        (det_bbox[..., 2:] - det_bbox[..., :2]) / 2 + det_bbox[..., :2])
            scale_ratio = torch.log(
                (trk_bbox[..., 2:] - trk_bbox[..., :2]) / (det_bbox[..., 2:] - det_bbox[..., :2]))  ###
        trkdet_rel = torch.concat((center_dis, scale_ratio), dim=-1)
        trkdet_emb = self.encoder(trkdet_rel)
        return trkdet_emb
