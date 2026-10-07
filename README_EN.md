# GujinBridge

GujinBridge is a domain language-model project for classical Chinese. It adapts the
[MedicalGPT](https://github.com/shibing624/MedicalGPT) training engine and adds a reproducible workflow for:

- classical Chinese to modern Chinese translation (`c2m`);
- modern Chinese to classical Chinese rewriting (`m2c`);
- sentence segmentation and punctuation (`punctuate`);
- source-grounded interpretation and retrieval-augmented question answering.

The primary documentation is the [Chinese README](README.md). It contains the complete data-preparation,
training, inference, evaluation, licensing, and attribution instructions.

## Current release candidate

The selected model stack is **Qwen3-1.7B -> V5 SFT merged -> DPO V1 LoRA**. On a source-isolated frozen set
of 1,440 examples, DPO improved overall character F1 from 72.69% to 73.39% and punctuation source-text
preservation from 64.54% to 71.19%. A targeted SFT run and two GRPO iterations were retained as controlled
ablations because they did not consistently outperform DPO.

The merged inference model is available on
[Hugging Face: tzh699/GujinBridge-Qwen3-1.7B-DPO](https://huggingface.co/tzh699/GujinBridge-Qwen3-1.7B-DPO).

![GujinBridge DPO V1 local demo](docs/assets/gujinbridge-demo.gif)

- [Final experiment report](docs/FINAL_EXPERIMENT_REPORT.md)
- [Training and post-training pipeline](docs/TRAINING_PIPELINE.md)
- [Model card](MODEL_CARD.md)

## Quick start

Install PyTorch for your CUDA environment first, then install the project dependencies:

```bash
pip install torch
pip install -r requirements.txt
```

Download and prepare the recommended corpus:

```bash
python tools/download_gujinbridge_dataset.py --output-dir data/gujinbridge/raw
python tools/prepare_gujinbridge_dataset.py \
  --input data/gujinbridge/raw \
  --output-dir data/gujinbridge/processed
```

Run single-GPU LoRA SFT:

```bash
bash scripts/run_sft_gujinbridge.sh
```

On Windows, run the selected local DPO model with:

```powershell
.\scripts\try_gujinbridge.ps1
```

For a pipeline smoke test, use the tiny in-repository demonstration data:

```bash
TRAIN_DIR=data/gujinbridge/demo/train \
VALIDATION_DIR=data/gujinbridge/demo/validation \
MAX_STEPS=2 \
bash scripts/run_sft_gujinbridge.sh
```

## Data and attribution

The recommended dataset is
[gujilab/chinese-classical-corpus](https://github.com/gujilab/chinese-classical-corpus). Full external data is not
vendored in this repository. Verify the exact upstream revision and license before use or redistribution.

The inherited framework and GujinBridge additions are distributed under Apache License 2.0. See
[NOTICE](NOTICE), [LICENSE](LICENSE), and [CITATION.cff](CITATION.cff) for attribution details.
