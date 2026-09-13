# AURORA: Training-Free Audio-Visual Routing for Mask Refinement and Reliability Assessment

AURORA is an inference-time method that uses frame-level predictions to refine video-level masks and re-score their confidence through query routing, mask refinement, and reliability estimation.

This repository contains the original **AVISM**, **ACVIS**, and **H2S** code
with AURORA integrated directly.

```text
AURORA/
  avis/     # AVISM + AURORA
  ACVIS/    # ACVIS + AURORA
  H2S/      # H2S + AURORA
```

## ⚙ Installation and data

Follow each model's original instructions for its environment, CUDA extensions,
AVISeg data, and pretrained checkpoints:
[AVISM](https://github.com/ruohaoguo/avis), [ACVIS](https://github.com/jinbae-s/ACVIS), [H2S](https://github.com/leiyeliu/H2S).
Run those instructions in the included model directory.

Use an **AVISeg-trained model checkpoint**, rather than a backbone initialization
checkpoint. Data and model weights must be obtained separately.

## 😺 Evaluation

Run each command from its model directory with the corresponding environment
active.

AVISM (`cd avis`):

```bash
python train_net.py --eval-only --num-gpus 1 \
  --config-file configs/avism/R50/avism_R50_IN_AURORA.yaml \
  MODEL.WEIGHTS /path/to/AVISM_R50_IN.pth
```

ACVIS (`cd ACVIS`):

```bash
python train_net.py --eval-only --num-gpus 1 \
  --config-file configs/acvis/acvis_R50_IN_AURORA.yaml \
  MODEL.WEIGHTS /path/to/ACVIS_R50_IN.pth
```

H2S (`cd H2S`):

```bash
python train_net.py --eval-only --num-gpus 1 \
  --config-file configs/h2s/R50/h2s_R50_IN_AURORA.yaml \
  MODEL.WEIGHTS /path/to/H2S_R50_IN.pth
```

| Model | AURORA preset directory | Backbone and pretraining |
|---|---|---|
| AVISM | `avis/configs/avism/` | R50-IN, R50-COCO, SwinL-IN |
| ACVIS | `ACVIS/configs/acvis/` | R50-IN, R50-COCO |
| H2S | `H2S/configs/h2s/` | R50-IN, R50-COCO |

Choose the corresponding `*_AURORA.yaml` and matching trained checkpoint.


## Acknowledgments

Please cite the original model papers when using their code. See [NOTICE.md](NOTICE.md) for upstream revisions and
license scope; the root [LICENSE](LICENSE) covers original AURORA contributions.
