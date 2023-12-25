import argparse


class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser("LTTrack parameters")
        self.parser.add_argument("--seed", default=None, type=int, help="eval seed")
        self.parser.add_argument("--num_works", default=12, type=int, help="number of workers")
        # det args
        self.parser.add_argument("--det_model_path", default="./weights/yolox/", type=str, help="detector path")
        self.parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
        self.parser.add_argument("--conf", default=0.1, type=float, help="test conf")
        self.parser.add_argument("--tsize", default=[800, 1440], nargs="+", type=int, help="test img size")

        # tracking args
        self.parser.add_argument("--track_thresh", type=float, default=0.6, help="detection confidence threshold")
        self.parser.add_argument(
            "--iou_thresh",
            type=float,
            default=0.3,
            help="the iou threshold in Sort for matching",
        )
        self.parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track in SORT")
        self.parser.add_argument("--min_box_area", type=float, default=10, help="filter out tiny boxes")
        self.parser.add_argument(
            "--aspect_ratio_thresh",
            type=float,
            default=1.6,
            help="threshold for filtering out boxes of which aspect ratio are above the given value.",
        )
        self.parser.add_argument(
            "--post",
            action="store_true",
            help="run post-processing linear interpolation.",
        )
        self.parser.add_argument("--w_assoc_emb", type=float, default=0.75, help="Combine weight for emb cost")
        self.parser.add_argument(
            "--alpha_fixed_emb",
            type=float,
            default=0.95,
            help="Alpha fixed for EMA embedding",
        )
        self.parser.add_argument("--emb_off", action="store_true")
        self.parser.add_argument("--aw_param", type=float, default=0.5)

        self.parser.add_argument(
            '--PBA',
            action='store_true',
            help='position cost association'
        )
        self.parser.add_argument("--assa_cfg", type=str, default=None)
        self.parser.add_argument("--assa_epoch", type=str, default=None)

        self.parser.add_argument(
            '--ZTRM',
            action='store_true',
            help='max-age cascade association'
        )
        self.parser.add_argument("--max_lost_age", type=int, default=30)
        self.parser.add_argument("--max_zombie_age", type=int, default=100)
        self.parser.add_argument("--ztrm_emb_thr", type=float, default=0.45)
        self.parser.add_argument("--ztrm_iou_thr", type=float, default=0.05)
        self.parser.add_argument("--motion_cfg", type=str, default=None)
        self.parser.add_argument("--motion_epoch", type=str, default=None)

        # eval args
        self.parser.add_argument("--result_folder", type=str, default="results/trackers/")
        self.parser.add_argument("--result_sub_folder", type=str, default="track_results")
        self.parser.add_argument("--output_sub_folder", type=str, default="", help="output sub folder for TrackEval")

        # exp args
        self.parser.add_argument("--dataset", type=str, default="mot17")
        self.parser.add_argument("--data_dir", type=str, default="/home/share/workspace/datasets")
        self.parser.add_argument("--test_dataset", action="store_true")
        self.parser.add_argument("--exp_name", type=str, default="exp1")


def make_parser():
    args = Args()
    return args.parser
