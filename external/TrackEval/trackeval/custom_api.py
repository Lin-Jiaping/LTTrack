import os
from .eval import Evaluator
from .datasets import MotChallenge2DBox
from .metrics import HOTA, CLEAR, Identity, VACE
# from trackeval import Evaluator
# from trackeval.datasets import MotChallenge2DBox
# from trackeval.metrics import HOTA, CLEAR, Identity, VACE


def set_trackeval_config(gt_folder="./external/TrackEval/trackeval_results/gt",
                         result_folder="results/trackers/",
                         exp_name="exp_name",
                         dataset="MOT17",
                         result_sub_folder="track_results",
                         output_sub_folder="",
                         split_to_eval="val",
                         metrics=None
                         ):
    code_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if metrics is None:
        metrics = ['HOTA', 'CLEAR', 'Identity']

    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 8,
        'BREAK_ON_ERROR': True,  # Raises exception and exits with error
        'RETURN_ON_ERROR': False,  # if not BREAK_ON_ERROR, then returns from function on error
        'LOG_ON_ERROR': os.path.join(code_path, 'error_log.txt'),  # if not None, save any errors into a log file.

        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'DISPLAY_LESS_PROGRESS': True,

        'OUTPUT_SUMMARY': True,
        'OUTPUT_EMPTY_CLASSES': True,  # If False, summary files are not output for classes with no detections
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': True,
    }

    dataset_config = {
        "GT_FOLDER": gt_folder,  # Location of GT data
        "TRACKERS_FOLDER": result_folder,  # Trackers location
        "OUTPUT_FOLDER": None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        "TRACKERS_TO_EVAL": [exp_name],  # Filenames of trackers to eval (if None, all in folder)
        "CLASSES_TO_EVAL": ["pedestrian"],  # Valid: ['pedestrian']
        "BENCHMARK": dataset.upper(),  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
        "SPLIT_TO_EVAL": split_to_eval,  # Valid: 'train', 'test', 'all'
        "INPUT_AS_ZIP": False,  # Whether tracker input files are zipped
        "PRINT_CONFIG": True,  # Whether to print current config
        "DO_PREPROC": True,  # Whether to perform preprocessing (never done for MOT15)
        "TRACKER_SUB_FOLDER": result_sub_folder,
        # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        "OUTPUT_SUB_FOLDER": output_sub_folder,
        # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
        "TRACKER_DISPLAY_NAMES": None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
        "SEQMAP_FOLDER": None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
        "SEQMAP_FILE": None,  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
        "SEQ_INFO": None,  # If not None, directly specify sequences to eval and their number of timesteps
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",  # '{gt_folder}/{seq}/gt/gt.txt'
        "SKIP_SPLIT_FOL": False,  # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
        # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
        # If True, then the middle 'benchmark-split' folder is skipped for both.
    }

    metric_config = {
        'METRICS': metrics,
        'THRESHOLD': 0.5
    }
    return eval_config, dataset_config, metric_config


def evaluate_trackeval(eval_config, dataset_config, metrics_config):
    # Run code
    evaluator = Evaluator(eval_config)
    dataset_list = [MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [HOTA, CLEAR, Identity, VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)


def gen_archive_structure(test_dir=None):
    if test_dir is None:
        test_dir = os.path.split(os.path.realpath(__file__))[0]
    test_list = ['MOT17-01-FRCNN.txt',
                 'MOT17-03-FRCNN.txt',
                 'MOT17-06-FRCNN.txt',
                 'MOT17-07-FRCNN.txt',
                 'MOT17-08-FRCNN.txt',
                 'MOT17-12-FRCNN.txt',
                 'MOT17-14-FRCNN.txt']

    detectors = ['DPM', 'SDP']

    for test_file in test_list:
        with open(os.path.join(test_dir, test_file), 'r') as fr:
            results = fr.read()
        for detector in detectors:
            filename = test_file.replace('FRCNN', detector)
            with open(os.path.join(test_dir, filename), 'w') as fw:
                fw.write(results)
