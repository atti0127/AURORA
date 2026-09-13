# Learning What To Hear: Boosting Sound-Source Association For Robust Audiovisual Instance Segmentation

[![ACVIS](https://img.shields.io/badge/Paper-ACVIS-red.svg?logo=arXiv)](https://arxiv.org/abs/2509.22740)
[![Web Demo](https://img.shields.io/badge/HuggingFace-Demo-yellow.svg?logo=huggingface)](https://huggingface.co/spaces/jinbae-s/ACVIS)

Jinbae Seo, Hyeongjun Kwon, Kwonyoung Kim, Jiyoung Lee and Kwanghoon Sohn

## Installation

```bash
conda create --name acvis python=3.8 -y
conda activate acvis

conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -U opencv-python
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install -r requirements.txt
pip install timm

cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

## Setup

### Datasets

Download and unzip [AVISeg datasets](https://ruohaoguo.github.io/avis/) and put them in `./datasets`.

### Pretrained Backbones

Download and unzip pre-trained backbones [OneDrive](https://1drv.ms/u/c/3c9af704fb61931d/ETDDliQ8zZFGmYxlLVPyi3sBis_fdjX0w8mJhyQnYVSdXA?e=Wt7pUb) and put them in `./pre_models`.

### Checkpoints

Download the following checkpoints and put them in `./checkpoints`.

| Backbone | Pre-trained Datasets | mAP | HOTA | FSLA | Model Weight |
| --- | --- | --- | --- | --- | --- |
| ResNet-50 | ImageNet | 42.01 | 62.04 | 42.43 | [ACVIS_R50_IN.pth](https://drive.google.com/file/d/1CpnXnMAFuoiQe1C-xWz4BHFXsuL3nFrh/view?usp=sharing) |
| ResNet-50 | ImageNet & COCO | 46.64 | 65.02 | 46.72 | [ACVIS_R50_COCO.pth](https://drive.google.com/file/d/1TXZg2nDtRyhAJijnG8SXv4aqyNlcl_nc/view?usp=sharing) |
