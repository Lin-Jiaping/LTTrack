import os
import torch
import yaml


class AssaEvaluator:
    def __init__(self, pos_cost_assa_cfg, assa_epoch, dataloader=None, device="cuda:0"):
        self.assa_epoch = assa_epoch
        self.dataloader = dataloader
        self.model = None
        self.device = device
        self.load_model(pos_cost_assa_cfg)

    def load_model(self, pos_cost_assa_cfg):
        from submodule.PBA.models.pos_cost_assa import AssaPosCost as AssaModel
        with open(pos_cost_assa_cfg, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        pos_cost_assa_dir = "./weights/pba"
        rel_pos_emb_dim = cfg["Assa Model"]["rel_pos_emb_dim"]
        trk_det_encoder_emb_dim = cfg["Assa Model"]["trk_det_encoder_emb_dim"]
        try:
            rel_k = cfg["Assa Model"]["rel_k"]
        except KeyError:
            rel_k = 4

        assa_ckpt_path = os.path.join(pos_cost_assa_dir, f"{self.assa_epoch}_ckpt.pth.tar")

        self.model = AssaModel(rel_pos_emb_dim=rel_pos_emb_dim,
                               trk_det_encoder_emb_dim=trk_det_encoder_emb_dim,
                               rel_k=rel_k,
                               mode='eval')

        ckpt = torch.load(assa_ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, trk_seq, trk_rel_pos, det_bbox, det_rel_pos):
        corr_matrix = self.model(trk_seq.to(self.device),
                                 trk_rel_pos.to(self.device),
                                 det_bbox.to(self.device),
                                 det_rel_pos.to(self.device))
        return corr_matrix
