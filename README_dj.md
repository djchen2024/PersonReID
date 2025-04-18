* SOLIDER
https://github.com/tinyvision/SOLIDER?tab=readme-ov-file ()
* Person Re-identification
https://github.com/tinyvision/SOLIDER-REID
* Person Search
* Pedestrian Detection
* Person Attribute Recognition
* Human Parsing
* Pose Estimation

### Installation

```
conda create -n solider python=3.10
conda activate solider
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
conda install -c conda-forge opencv -n solider
pip install yacs timm scikit-image tqdm ftfy regex 
pip install numpy==1.24
```
<!-- @swin_transformer.py  >> from mmengine.runner import load_checkpoint as _load_checkpoint # due to mmcv version -->
<!-- @vit_pytorch.py  >>  import collections.abc as container_abcs # DJ -->

### Datasets
* Market-1501
```
pip install gdown
gdown https://drive.google.com/uc\?id\=0B8-rUzbwVRk0c054eEozWG9COHM
```
 
### Testing set
```
/nfs/ssd13/test_set/vision/reid
```

### Backbones / Pre-trained Models
* Swin Tiny (811M)
```
gdown 12UyPVFmjoMVpQLHN07tNh4liHUmyDqg8
```
* Swin Small (1.15G)
```
gdown 1oyEgASqDHc7YUPsQUMxuo2kBZyi2Tzfv
```
* Swin Base (1.77G)
```
gdown 1uh7tO34tMf73MJfFqyFEGx42UBktTbZU
```


### Trained Models @ Market1501 (w/o RK)
* Swin Tiny (113M)
```
gdown 1YrE_r9Fk5uR0uFFQboBv203vxlOpFXU8
```
* Swin Small (198M)
```
gdown 14uOCf5yZq0Rt5rRSJI9I7_d5kt2EOyHO
```
* Swin Base (351M)
```
gdown 1pZ1unW2IwSsqSN2KYHcgBhgjQztQW_fe
```


### Trained Models @ MSMT17 (w/o RK)
* Swin Tiny (114M)
```
gdown 10YLhMbwvmxZl3gTVo2BN_828SKZHdCjr
```
* Swin Small (199M)
```
gdown 1C-aIZdFyjFsZX4W4feG-Ex39RU2Qvu3b
```
* Swin Base (352M)
```
gdown 1Y-RFAYdT56vnMjwxH1Ym3DVhZzZuMQZs
```

### Train
```
sh run.sh
```

### Test
```
sh runtest.sh
```

















[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-appearance-a-semantic-controllable/pedestrian-attribute-recognition-on-pa-100k)](https://paperswithcode.com/sota/pedestrian-attribute-recognition-on-pa-100k?p=beyond-appearance-a-semantic-controllable)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-appearance-a-semantic-controllable/person-re-identification-on-msmt17)](https://paperswithcode.com/sota/person-re-identification-on-msmt17?p=beyond-appearance-a-semantic-controllable)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-appearance-a-semantic-controllable/person-re-identification-on-market-1501)](https://paperswithcode.com/sota/person-re-identification-on-market-1501?p=beyond-appearance-a-semantic-controllable)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-appearance-a-semantic-controllable/person-search-on-cuhk-sysu)](https://paperswithcode.com/sota/person-search-on-cuhk-sysu?p=beyond-appearance-a-semantic-controllable)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-appearance-a-semantic-controllable/person-search-on-prw)](https://paperswithcode.com/sota/person-search-on-prw?p=beyond-appearance-a-semantic-controllable)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-appearance-a-semantic-controllable/pedestrian-detection-on-citypersons)](https://paperswithcode.com/sota/pedestrian-detection-on-citypersons?p=beyond-appearance-a-semantic-controllable)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-appearance-a-semantic-controllable/semantic-segmentation-on-lip-val)](https://paperswithcode.com/sota/semantic-segmentation-on-lip-val?p=beyond-appearance-a-semantic-controllable)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-appearance-a-semantic-controllable/pose-estimation-on-coco)](https://paperswithcode.com/sota/pose-estimation-on-coco?p=beyond-appearance-a-semantic-controllable)

Welcome to **SOLIDER**! SOLIDER is a Semantic Controllable Self-Supervised Learning Framework to learn general human representations from massive unlabeled human images which can benefit downstream human-centric tasks to the maximum extent. Unlike the existing self-supervised learning methods, prior knowledge from human images is utilized in SOLIDER to build pseudo semantic labels and import more semantic information into the learned representation. Meanwhile, different downstream tasks always require different ratios of semantic information and appearance information, and a single learned representation cannot fit for all requirements. To solve this problem, SOLIDER introduces a conditional network with a semantic controller, which can fit different needs of downstream tasks. For more details, please refer to our paper [Beyond Appearance: a Semantic Controllable Self-Supervised Learning Framework for Human-Centric Visual Tasks](https://arxiv.org/abs/2303.17602).

<div align="center"><img src="assets/framework.png" width="900"></div>

## Updates
- **[2023/07/21: Codes of human pose task is released!] ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png)**
    * Training details of our pretrained model on downstream human pose task is released.
- **[2023/05/15: Codes of human parsing task is released!] ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png)**
    * Training details of our pretrained model on downstream human parsing task is released.
- **[2023/04/24: Codes of attribute recognition task is released!] ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png)**
    * Training details of our pretrained model on downstream person attribute recognition task is released.
- **[2023/03/28: Codes of 3 downstream tasks are released!]**
    * Training details of our pretrained model on 3 downstream human visual tasks, including person re-identification, person search and pedestrian detection, are released.
- **[2023/03/13: SOLIDER is accepted by CVPR2023!]**
    * The paper of SOLIDER is accepted by CVPR2023, and its offical pytorch implementation is released in this repo. 

## Installation
This codebase has been developed with python version 3.7, PyTorch version 1.7.1, CUDA 10.1 and torchvision 0.8.2.                                           

## Datasets
We use **LUPerson** as our training data, which consists of unlabeled human images. Download **LUPerson** from its [offical link](https://github.com/DengpanFu/LUPerson) and unzip it.

## Training
- Choice 1. To train SOLIDER from scratch, please run:
```shell
sh run_solider.sh
```

- Choice 2. Training SOLIDER from scratch may take a long time. To speed up the training, you can train a DINO model first, and then finetune it with SOLIDER, as follows:
```shell
sh run_dino.sh
sh resume_solider.sh
```

## Finetuning and Inference
There is a demo to run the trained SOLIDER model, which can be embedded into the inference or the downstream task finetuning.
```shell
python demo.py
```

## Models
We use [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) as our backbone, which shows great advantages on many CV tasks.
| Task | Dataset | Swin Tiny<br>([Link](https://drive.google.com/file/d/12UyPVFmjoMVpQLHN07tNh4liHUmyDqg8/view?usp=share_link)) | Swin Small<br>([Link](https://drive.google.com/file/d/1oyEgASqDHc7YUPsQUMxuo2kBZyi2Tzfv/view?usp=share_link)) | Swin Base<br>([Link](https://drive.google.com/file/d/1uh7tO34tMf73MJfFqyFEGx42UBktTbZU/view?usp=share_link)) |
| :---: |:---: |:---: | :---: | :---: |
| Person Re-identification (mAP/R1)<br>w/o re-ranking | Market1501 | 91.6/96.1 | 93.3/96.6 | 93.9/96.9 |
|  | MSMT17 | 67.4/85.9 | 76.9/90.8 | 77.1/90.7 |
| Person Re-identification (mAP/R1)<br>with re-ranking | Market1501 | 95.3/96.6 | 95.4/96.4 | 95.6/96.7 |
|  | MSMT17 | 81.5/89.2 | 86.5/91.7 | 86.5/91.7 |
| Attribute Recognition (mA) | PETA_ZS | 74.37 | 76.21 | 76.43 |
|  | RAP_ZS | 74.23 | 75.95 | 76.42 |
|  | PA100K | 84.14 | 86.25 | 86.37 |
| Person Search (mAP/R1) | CUHK-SYSU | 94.9/95.7 | 95.5/95.8 | 94.9/95.5 |
|  | PRW | 56.8/86.8 | 59.8/86.7 | 59.7/86.8 |
| Pedestrian Detection (MR-2) | CityPersons | 10.3/40.8 | 10.0/39.2 | 9.7/39.4 |
| Human Parsing (mIOU) | LIP | 57.52 | 60.21 | 60.50 |
| Pose Estimation (AP/AR) | COCO | 74.4/79.6 | 76.3/81.3 | 76.6/81.5 |

- All the models are trained on the whole LUPerson dataset.

## Traning codes on Downstream Tasks
- [Person Re-identification](https://github.com/tinyvision/SOLIDER-REID)
- [Person Search](https://github.com/tinyvision/SOLIDER-PersonSearch)
- [Pedestrian Detection](https://github.com/tinyvision/SOLIDER-PedestrianDetection)
- [Person Attribute Recognition](https://github.com/tinyvision/SOLIDER-PersonAttributeRecognition)
- [Human Parsing](https://github.com/tinyvision/SOLIDER-HumanParsing)
- [Pose Estimation](https://github.com/tinyvision/SOLIDER-HumanPose)

## Acknowledgement
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.
- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- [DINO](https://github.com/facebookresearch/dino)
- [TransReID](https://github.com/damo-cv/TransReID)
- [TransReID-SSL](https://github.com/damo-cv/TransReID-SSL)
- [SeqNet](https://github.com/serend1p1ty/SeqNet)
- [Pedestron](https://github.com/hasanirtiza/Pedestron)
- [LUPerson](https://github.com/DengpanFu/LUPerson)
- [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)
- [mmpose](https://github.com/open-mmlab/mmpose)

## Reference
If you use SOLIDER in your research, please cite our work by using the following BibTeX entry:
```
@inproceedings{chen2023beyond,
  title={Beyond Appearance: a Semantic Controllable Self-Supervised Learning Framework for Human-Centric Visual Tasks},
  author={Weihua Chen and Xianzhe Xu and Jian Jia and Hao Luo and Yaohua Wang and Fan Wang and Rong Jin and Xiuyu Sun},
  booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023},
}
