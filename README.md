# AURORA: Training-Free Audio-Visual Routing for Mask Refinement and Reliability Assessment

Training-free query routing, mask refinement, and prediction re-scoring for
audio-visual instance segmentation.

This repository contains the original **AVISM**, **ACVIS**, and **H2S** code
with AURORA integrated directly.

```text
AURORA/
  avis/     # AVISM + AURORA
  ACVIS/    # ACVIS + AURORA
  H2S/      # H2S + AURORA
```

## Installation and data

Follow each model's original instructions for its environment, CUDA extensions,
AVISeg data, and pretrained checkpoints:
[AVISM](https://github.com/ruohaoguo/avis), [ACVIS](https://github.com/jinbae-s/ACVIS), [H2S](https://github.com/leiyeliu/H2S).
Run those instructions in the included model directory.

Use an **AVISeg-trained model checkpoint**, rather than a backbone initialization
checkpoint. Data and model weights must be obtained separately.

## Evaluation

Run each command from its model directory with the corresponding environment
active. Set `DETECTRON2_DATASETS` to the AVISeg root containing `test.json` and
`test/`, or prepare data in that model directory's `datasets/` folder.

AVISM (`cd avis`):

```bash
CUDA_VISIBLE_DEVICES=0 DETECTRON2_DATASETS=/path/to/AVISeg \
python train_net.py --eval-only --num-gpus 1 \
  --config-file configs/avism/R50/avism_R50_IN_AURORA.yaml \
  MODEL.WEIGHTS /path/to/AVISM_R50_IN.pth
```

ACVIS (`cd ACVIS`):

```bash
CUDA_VISIBLE_DEVICES=0 DETECTRON2_DATASETS=/path/to/AVISeg \
python train_net.py --eval-only --num-gpus 1 \
  --config-file configs/acvis/acvis_R50_IN_AURORA.yaml \
  MODEL.WEIGHTS /path/to/ACVIS_R50_IN.pth
```

H2S (`cd H2S`):

```bash
CUDA_VISIBLE_DEVICES=0 DETECTRON2_DATASETS=/path/to/AVISeg \
python train_net.py --eval-only --num-gpus 1 \
  --config-file configs/h2s/R50/h2s_R50_IN_AURORA.yaml \
  MODEL.WEIGHTS /path/to/H2S_R50_IN.pth
```

For the original baseline, append `MODEL.AVISM.AURORA_ENABLED False` and a
separate `OUTPUT_DIR /path/to/baseline_output` to the same command.

| Model | AURORA preset directory | Backbone and pretraining |
|---|---|---|
| AVISM | `avis/configs/avism/` | R50-IN, R50-COCO, SwinL-IN, SwinL-COCO |
| ACVIS | `ACVIS/configs/acvis/` | R50-IN, R50-COCO |
| H2S | `H2S/configs/h2s/` | R50-IN, R50-COCO, SwinL-COCO |

Choose the corresponding `*_AURORA.yaml` and matching trained checkpoint.

## AURORA implementation

The default uses embedding-based query routing, foreground removal and logit
averaging for mask refinement, and Dice agreement between the original masks
for reliability estimation. It requires no retraining or additional parameters.

The helpers are in `avis/avism/modeling/aurora.py`,
`ACVIS/models/modeling/aurora.py`, and `H2S/h2s/modeling/aurora.py`.
They are integrated into each model's native inference code. The
`MODEL.AVISM.AURORA_ENABLED` flag selects AURORA; it is enabled by the AURORA
presets and disabled by default in the original configurations.

## Acknowledgments

Please cite the original model papers when using their code. Their citations
and documentation remain in the model READMEs. Original licenses and copyright
notices are preserved. See [NOTICE.md](NOTICE.md) for upstream revisions and
license scope; the root [LICENSE](LICENSE) covers original AURORA contributions.
