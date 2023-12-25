import os
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random

from data import dataset
import utils
from external.adaptors import detector

from trackers import long_term_track as tracker_module
from trackers.long_term_track.long_term_tracker import LongTermMotionTracker as Tracker
from external.TrackEval.trackeval.custom_api import set_trackeval_config, evaluate_trackeval


def get_main_args():
    parser = tracker_module.args.make_parser()
    args = parser.parse_args()

    if "mot17" in args.dataset.lower():
        args.result_folder = os.path.join(args.result_folder, "MOT17-val")
    elif args.dataset == "mot20":
        args.result_folder = os.path.join(args.result_folder, "MOT20-val")
    elif args.dataset == "dance":
        args.result_folder = os.path.join(args.result_folder, "DANCE-val")
    if args.test_dataset:
        args.result_folder = args.result_folder.replace("-val", "-test")
    args.result_sub_folder = os.path.join(args.output_sub_folder, args.result_sub_folder)

    args.seed=6
    return args


def get_detector(args):
    det_model_path = args.det_model_path
    det = None
    if "mot17" in args.dataset.lower():
        if args.test_dataset:
            detector_path = os.path.join(det_model_path, "bytetrack_x_mot17.pth.tar")
        else:
            detector_path = os.path.join(det_model_path, "bytetrack_ablation.pth.tar")
        size = (800, 1440)
    elif args.dataset == "mot20":
        if args.test_dataset:
            detector_path = os.path.join(det_model_path, "bytetrack_x_mot20.tar")
            size = (896, 1600)
        else:
            detector_path = os.path.join(det_model_path, "bytetrack_x_mot17.pth.tar")
            size = (800, 1440)
    elif args.dataset == "dance":
        detector_path = os.path.join(det_model_path, "bytetrack_dance_model.pth.tar")
        size = (800, 1440)
    else:
        raise RuntimeError("Need to update paths for detector for extra datasets.")

    det = detector.Detector("yolox", detector_path, args.dataset)
    return det, size


def set_tracker_attr(tracker, **options):
    for opt in options.keys():
        if options[opt] is not None:
            setattr(tracker, opt, options[opt])


def evaluator(args, trk_options):
    det, size = get_detector(args)
    loader = dataset.get_mot_loader(args.dataset,
                                    args.test_dataset,
                                    data_dir=args.data_dir,
                                    size=size,
                                    workers=args.num_works)
    # Set up tracker
    tracker_args = dict(
        args=args,
        det_thresh=args.track_thresh,
        max_age=args.max_lost_age,
        iou_threshold=args.iou_thresh,
        w_association_emb=args.w_assoc_emb,
        alpha_fixed_emb=args.alpha_fixed_emb,
        embedding_off=args.emb_off,
        aw_param=args.aw_param,
    )
    set_tracker_attr(Tracker, **trk_options)
    tracker = Tracker(**tracker_args)
    Tracker.load_motion_model()
    if "PBA" in args.__dict__.keys() and args.PBA:
        Tracker.load_assa_model()
    results = {}
    frame_count = 0
    total_time = 0
    for (img, np_img), label, info, idx in loader:
        frame_id = info[2].item()  # start from 1
        video_name = info[4][0].split("/")[0]

        if "FRCNN" not in video_name and args.dataset == "mot17":
            continue
        tag = f"{video_name}:{frame_id}"
        if video_name not in results:
            results[video_name] = []
        img = img.cuda()

        print(f"Processing {video_name}:{frame_id}\r", end="")
        if frame_id == 1:
            print(f"Initializing tracker for {video_name}")
            print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
            tracker.dump_cache()
            tracker = Tracker(**tracker_args)

        start_time = time.time()
        pred = det(img, tag)
        if pred is None:
            continue
        targets = tracker.update(pred.clone(), img, np_img[0].numpy(), tag)
        tlwhs, ids = utils.filter_targets(targets, args.aspect_ratio_thresh, args.min_box_area)

        total_time += time.time() - start_time
        frame_count += 1

        results[video_name].append((frame_id, tlwhs, ids))

    print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
    det.dump_cache()
    tracker.dump_cache()

    folder = os.path.join(args.result_folder, args.exp_name, args.result_sub_folder)
    os.makedirs(folder, exist_ok=True)
    for name, res in results.items():
        result_filename = os.path.join(folder, f"{name}.txt")
        utils.write_results_no_score(result_filename, res)
    print(f"Finished, results saved to {folder}")
    if args.post:
        post_folder = os.path.join(args.result_folder, args.exp_name + "_post")
        pre_folder = os.path.join(args.result_folder, args.exp_name)
        if os.path.exists(post_folder):
            print(f"Overwriting previous results in {post_folder}")
            shutil.rmtree(post_folder)
        shutil.copytree(pre_folder, post_folder)
        post_folder_data = os.path.join(post_folder, args.result_sub_folder)
        utils.dti(post_folder_data, post_folder_data)
        print(f"Linear interpolation post-processing applied, saved to {post_folder_data}.")

    if not args.test_dataset:
        eval_config, dataset_config, metrics_config = set_trackeval_config(
            result_folder=os.path.split(args.result_folder)[0],
            exp_name=args.exp_name,
            dataset=args.dataset,
            result_sub_folder=args.result_sub_folder,
            output_sub_folder=args.output_sub_folder,
            split_to_eval="val")
        evaluate_trackeval(eval_config, dataset_config, metrics_config)


def main():
    np.set_printoptions(suppress=True, precision=5)
    args = get_main_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    if "motion_epoch" in args.__dict__.keys():
        trk_options = dict(
            motion_model_cfg=args.motion_cfg,
            motion_epoch=args.motion_epoch,
            assa_model_cfg=args.assa_cfg,
            assa_epoch=args.assa_epoch,
        )
    else:
        trk_options = dict()
    evaluator(args, trk_options)


if __name__ == "__main__":
    main()
