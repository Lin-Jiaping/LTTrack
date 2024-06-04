# LTTrack

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![test](https://img.shields.io/static/v1?label=By&message=Pytorch&color=red)

This is the official repository of "[LTTrack: Rethinking the Tracking Framework for Long-Term Multi-Object Tracking](https://ieeexplore.ieee.org/abstract/document/10536914)".

<table>
<tr>
    <td><img src="assets/MOT17-01.gif"></td>
    <td><img src="assets/MOT17-03.gif"></td>
    <td><img src="assets/MOT17-07.gif"></td>
</tr>
<tr>
    <td><img src="assets/MOT17-08.gif"></td>
    <td><img src="assets/MOT17-12.gif"></td>
    <td><img src="assets/MOT17-14.gif"></td>
</tr>
</table>

## Setup
To set up this repository follow the following steps:
1. Clone this repository and install external dependencies.
```
git clone https://github.com/Lin-Jiaping/LTTrack.git

cd LTTrack/external/YOLOX
python setup.py develop

cd ../../external/fast_reid
python setup.py develop

cd ../../external/TrackEval
python setup.py develop

cd ../../
pip install -r requirements.txt

```

2. Download the [weights](https://drive.google.com/drive/folders/1Rw2V5oM-YSVZw9OJzzQj5unpoa8FP8HN?usp=sharing) and add them to `LTTrack/weights` directory.
3. Download [MOT17](https://motchallenge.net/data/MOT17/), [MOT20](https://motchallenge.net/data/MOT20/), and [DanceTrack](https://github.com/DanceTrack/DanceTrack) datasets. The expected folder structure is:
 ```
 datasets
 |——————mot
 |        └——————train
 |        └——————test
 └——————MOT20
 |        └——————train
 |        └——————test
 └——————dancetrack        
          └——————train
          └——————val
          └——————test
 ```
4. Turn the datasets to COCO format.
```
# change paths of datasets in the file first.
python tools/data/convert_dance_to_coco.py
python tools/data/convert_mot17_to_coco.py
python tools/data/convert_mot20_to_coco.py
```

## Tracking
Run LTTrack on DanceTrack:
```
python main.py --exp_name DANCE-test --dataset mot20 --test_dataset --PBA --ZTRM --motion_cfg ./weights/ltm/ltm_dance.yml --motion_epoch ltm_dance --assa_cfg ./weights/pba/pba_dance.yml --assa_epoch pba_dance
```

Run LTTrack on MOT17:
```
python main.py --exp_name MOT17-test --dataset mot17 --test_dataset --PBA --ZTRM --ztrm_iou_thr 0.15 --max_lost_age 20 --max_zombie_age 130 --post --motion_cfg ./weights/ltm/ltm_mot17.yml --motion_epoch ltm_mot17 --assa_cfg ./weights/pba/pba_mot17.yml --assa_epoch pba_mot17
```

Run LTTrack on MOT20:
```
python main.py --exp_name MOT20-test --dataset mot20 --test_dataset --track_thresh 0.4 --PBA --ZTRM --ztrm_emb_thr 0.6 --post --motion_cfg ./weights/ltm/ltm_mot20.yml --motion_epoch ltm_mot20 --assa_cfg ./weights/pba/pba_mot20.yml --assa_epoch pba_mot20
```

## Acknowledgements
The code base is built upon [OC-SORT](https://github.com/noahcao/OC_SORT), [DeepOC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [FastReID](https://github.com/JDAI-CV/fast-reid), [TrackEval](https://github.com/JonathonLuiten/TrackEval). We thank the authors for their wonderful works!

## Citation
```
@ARTICLE{lttrack,
  author={Lin, Jiaping and Liang, Gang and Zhang, Rongchuan},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={LTTrack: Rethinking the Tracking Framework for Long-Term Multi-Object Tracking}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Target tracking;Predictive models;Feature extraction;Computational modeling;Market research;Transformers;Data models;Multi-object tracking;long-term tracking;tracking-by-detection;motion model;data association},
  doi={10.1109/TCSVT.2024.3404275}}
```