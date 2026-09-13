# Audio-Visual Instance Segmentation


[![AVIS](https://img.shields.io/badge/Paper-AVIS-2b9348.svg?logo=arXiv)](https://arxiv.org/abs/2310.18709)
[![Project Page](https://img.shields.io/badge/Project_page-Visualizations-blue)](https://ruohaoguo.github.io/avis/)
[![Dataset](https://img.shields.io/badge/Dataset-Download-yellow)](https://1drv.ms/u/c/3c9af704fb61931d/EVOs609SGMxLsbvVzVJHAa4Bmnu4GVZGjqYHQxDz0NKTew?e=WQU2Uf)

Ruohao Guo, Xianghua Ying*, Yaru Chen, Dantong Niu, Guangyao Li, Liao Qu, Yanyu Qi, Jinxing Zhou, Bowei Xing, Wenzhen Yue, Ji Shi, Qixun Wang, Peiliang Zhang, Buwen Liang

## 📰 News

🔥**2025.03.01**: Code and checkpoints are released!

🔥**2025.02.27**: AVIS got accepted to **CVPR 2025**! 🎉🎉🎉

🔥**2024.11.12**: Our [project page](https://ruohaoguo.github.io/avis/) is now available!

🔥**2024.11.11**: The AVISeg dataset has been uploaded to [OneDrive](https://1drv.ms/u/c/3c9af704fb61931d/EVOs609SGMxLsbvVzVJHAa4Bmnu4GVZGjqYHQxDz0NKTew?e=WQU2Uf), welcome to download and use!


## 🌿 Introduction

In this paper, we propose a new multi-modal task, termed audio-visual instance segmentation (AVIS), which aims to simultaneously identify, segment and track individual sounding object instances in audible videos. To facilitate this research, we introduce a high-quality benchmark named AVISeg, containing over 90K instance masks from 26 semantic categories in 926 long videos. Additionally, we propose a strong baseline model for this task. Our model first localizes sound source within each frame, and condenses object-specific contexts into concise tokens. Then it builds long-range audio-visual dependencies between these tokens using window-based attention, and tracks sounding objects among the entire video sequences.

<div align='center'>
<img src="./assets/teaser_figure.png" class="interpolation-image" alt="radar." height="50%" width="100%" />
</div>



## ⚙️ Installation

```bash
conda create --name avism python=3.8 -y
conda activate avism

conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

cd ./AVISM
git clone https://github.com/facebookresearch/detectron2
cd detectron2
pip install -e .

cd ../
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

## 🤗 Setup

### Datasets

Download and unzip datasets [OneDrive](https://1drv.ms/u/c/3c9af704fb61931d/EVOs609SGMxLsbvVzVJHAa4Bmnu4GVZGjqYHQxDz0NKTew?e=WQU2Uf) and put them in ```./datasets```.

### Pretrained Backbones
Download and unzip pre-trained backbones [OneDrive](https://1drv.ms/u/c/3c9af704fb61931d/ETDDliQ8zZFGmYxlLVPyi3sBis_fdjX0w8mJhyQnYVSdXA?e=Wt7pUb) and put them in ```./pre_models```.

### Checkpoints

Download the following checkpoints and put them in ```./checkpoints```.

<table>
  <tr>
    <th style="width: 150px;">Backbone</th>
    <th>Pre-trained Datasets</th>
    <th>FSLA</th>
    <th>HOTA</th>
    <th>mAP</th>
    <th>Model Weight</th>
  </tr>
  <tr>
    <td align="center">ResNet-50</td>
    <td align="center">ImageNet</td>
    <td align="center">42.78</td>
    <td align="center">61.73</td>
    <td align="center">40.57</td>
    <td align="center"><a href="https://1drv.ms/u/c/3c9af704fb61931d/EYyAuCNpRjxDqEohJfoDLO0BYgw0lbwKqQ1lwVXe_kIPVQ?e=PeRlyx">AVISM_R50_IN.pth</a></td>
  </tr>
  <tr>
    <td align="center">ResNet-50</td>
    <td align="center">ImageNet & COCO</td>
    <td align="center">44.42</td>
    <td align="center">64.52</td>
    <td align="center">45.04</td>
    <td align="center"><a href="https://1drv.ms/u/c/3c9af704fb61931d/EX0snZsxQwdBswQFdG4sc9kBd-Bd7lw5zaTGR6FvrSxinQ?e=bdZF5G">AVISM_R50_COCO.pth</a></td>
  </tr>
  <tr>
    <td align="center">Swin-L</td>
    <td align="center">ImageNet</td>
    <td align="center">49.15</td>
    <td align="center">68.81</td>
    <td align="center">49.06</td>
    <td align="center"><a href="https://1drv.ms/u/c/3c9af704fb61931d/EV4V5Bh5AqVBhLVMM1ucdN0BuOZgHu17W3JDGjKDMLZ1bg?e=hF8umh">AVISM_SwinL_IN.pth</a></td>
  </tr>
  <tr>
    <td align="center">Swin-L</td>
    <td align="center">ImageNet & COCO</td>
    <td align="center">52.49</td>
    <td align="center">71.13</td>
    <td align="center">53.46</td>
    <td align="center"><a href="https://1drv.ms/u/c/3c9af704fb61931d/EXuM4cUxPTpEk1M7FoPqtNEBi47L7uR-ZlnqDCJscmNsiA?e=7prFiN">AVISM_SwinL_COCO.pth</a></td>
  </tr>
</table>


## 📌 Getting Started

### Training
```
python train_net.py --num-gpus 2 --config-file configs/avism/R50/avism_R50_IN.yaml
```

### Evaluation
```
python train_net.py --config-file configs/avism/R50/avism_R50_IN.yaml --eval-only MODEL.WEIGHTS checkpoints/AVISM_R50_IN.pth
```

### Demo
```
python demo_video/demo.py --config-file configs/avism/R50/avism_R50_IN.yaml --opts MODEL.WEIGHTS checkpoints/AVISM_R50_IN.pth
```

## Acknowledgement

We thank the great work from [Detectron2](https://github.com/facebookresearch/detectron2), [Mask2Former](https://github.com/facebookresearch/MaskFormer) and [VITA](https://github.com/sukjunhwang/VITA).


## 📄 Citation

If our work assists your research, feel free to give us a star ⭐ or cite us using

```
@article{guo2023audio,
  title={Audio-Visual Instance Segmentation},
  author={Guo, Ruohao and Ying, Xianghua and Chen, Yaru and Niu, Dantong and Li, Guangyao and Qu, Liao and Qi, Yanyu and Zhou, Jinxing and Xing, Bowei and Yue, Wenzhen and Shi, Ji and Wang, Qixun and Zhang, Peiliang and Liang, Buwen},
  journal={arXiv preprint arXiv:2310.18709},
  year={2023}
}
```
