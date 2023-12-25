import os
import torch
import yaml


class MotionEvaluator:
    def __init__(self, motion_model_cfg, motion_epoch, dataloader=None, device="cuda:0"):
        self.dataloader = dataloader
        self.device = device
        self.load_model(motion_model_cfg, motion_epoch)

    def load_model(self, motion_model_cfg, motion_epoch):
        from submodule.LTM.models import LongTermMotion as MotionModel
        with open(motion_model_cfg, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        motion_model_dir = "./weights/ltm"
        motion_model_path = os.path.join(motion_model_dir, f"{motion_epoch}_ckpt.pth.tar")

        z_size = cfg["Motion Model"]["noise_dim"]
        inp_format = cfg["Motion Model"]["inp_format"]
        encoder_h_dim = cfg["Motion Model"]["h_dim"]
        decoder_h_dim = cfg["Motion Model"]["decoder_h_dim"]
        social_feat_size = cfg["Motion Model"]["h_dim"]
        embedding_dim = int(cfg["Motion Model"]["decoder_h_dim"] // 2)
        pred_len = cfg["Motion Model"]["pred_len"]
        try:
            long_term_emb_dim = cfg["Motion Model"]["long_term_emb_dim"]
            max_input_len = cfg["Motion Model"]["max_input_len"]
        except KeyError:
            long_term_emb_dim = 32
            max_input_len = 30

        self.model = MotionModel(
            z_size=z_size,
            inp_format=inp_format,
            encoder_h_dim=encoder_h_dim,
            decoder_h_dim=decoder_h_dim,
            social_feat_size=social_feat_size,
            embedding_dim=embedding_dim,
            long_term_emb_dim=long_term_emb_dim,
            pred_len=pred_len,
            max_inp_len=max_input_len
        )
        ckpt = torch.load(motion_model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, in_xywh, in_dxdydwdh):
        """
        Args: (batch_size = track_num)
            - in_xywh: (track_num, seq_len, 4)
            - in_dxdydwdh: (track_num, seq_len, 4)
        """
        pred = self.model(in_xywh, in_dxdydwdh)
        return pred.abs  # (track_num, 4)
