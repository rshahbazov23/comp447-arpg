# Random Token Rejection for Parallel Autoregressive Image Generation

Random Token Rejection (RTR) is a change to the decoding loop of [ARPG](https://github.com/hp-l33/ARPG) (Li et al., ICLR 2026), an autoregressive image generator that predicts many tokens in parallel at each step. At every step, RTR holds back a uniformly random fraction of the tokens just predicted and predicts them again at a later step, when more of the image is already in the context. It needs no retraining and runs on the released ARPG checkpoints.

With 8 decoding steps, RTR lowers FID-50K on ImageNet 256×256 from 10.51 to 3.62 for ARPG-L, at under 5% extra wall-clock time.

This was the course project for Deep Unsupervised Learning (COMP 447) at Koç University in Spring 2026, by Riad Shahbazov, Ömer Maraş and Mohamad Alomar. The code is built on the official ARPG implementation; see [Credits](#credits).

## How it works

ARPG fills in an image's tokens in random order over a fixed number of steps. Each step predicts a group of tokens independently of one another and commits all of them to the key-value cache. A poor prediction made early, with little context, therefore stays in the context of every later step.

RTR adds one rule. After each step, a fraction ρ of that step's tokens, chosen uniformly at random, is not committed. Those positions go back into the queue and are predicted again in a later step. The last step commits everything, so the number of steps does not change. ρ is the only hyperparameter, and all results below use ρ = 0.7.

The project began as confidence-guided rejection: defer the tokens the model is least sure of. A control run that deferred random tokens instead did better than every confidence measure we tried, so the random rule became the method and the confidence rules became the ablation.

## Results

Class-conditional ImageNet 256×256, FID-50K measured with the OpenAI ADM evaluator, released checkpoints, arccos sampling schedule. Lower is better.

**ARPG-L** (320M parameters, CFG 5.0, mean of 5 seeds)

| Steps | ARPG | ARPG + RTR | Change |
|---:|---:|---:|---:|
| 8 | 10.51 | 3.62 | −65.6% |
| 12 | 5.29 | 2.79 | −47.2% |
| 16 | 3.57 | 2.60 | −27.3% |
| 24 | 2.58 | 2.54 | −1.7% |
| 32 | 2.38 | 2.57 | +7.7% |

**ARPG-XL** (719M parameters, CFG 6.0, mean of 3 seeds, ρ carried over from ARPG-L without retuning)

| Steps | ARPG | ARPG + RTR | Change |
|---:|---:|---:|---:|
| 8 | 11.82 | 4.04 | −65.8% |
| 16 | 3.78 | 2.43 | −35.7% |
| 32 | 2.19 | 2.18 | −0.4% |

What the experiments show:

- **RTR is a step-efficiency method.** It helps most when steps are few. At 16 steps it recovers 82% of the FID gap between 16-step and 32-step ARPG-L, and 85% for ARPG-XL. At 24 steps it makes no real difference, and at 32 steps it makes ARPG-L worse.
- **Random selection beats confidence.** At 8 steps and the same ρ, random deferral reaches FID 3.63, against 6.69 for the top-1 to top-2 margin, 7.70 for maximum probability and 7.73 for entropy (3 seeds each).
- **Deferring early is what helps.** A schedule that defers little in early steps and a lot in late ones raises 8-step FID-10K from 6.10 to 9.18.
- **The cost is small.** Wall-clock overhead is 4.7% at 8 steps, 3.4% at 16 and 1.1% at 32, measured on one A100.
- **It trades recall for precision.** Precision rises from about 0.78 to 0.81 and recall falls from about 0.60 to 0.56.

Before changing anything we reproduced the published ARPG-L result: FID 2.42 at 64 steps, against 2.37 in the paper.

Every run, with per-seed numbers, is in [docs/rtr_experiments_log.md](docs/rtr_experiments_log.md). The earlier confidence-guided experiments are in [docs/experiments_log.md](docs/experiments_log.md).

## Running it

Sampling needs a Linux machine with an NVIDIA GPU. [docs/ARPG_L_BASELINE.md](docs/ARPG_L_BASELINE.md) covers the environment, the downloads and the FID evaluation step by step.

```sh
# Install CUDA-enabled PyTorch first, then:
python -m pip install -r requirements-repro.txt
./scripts/download_baseline_assets.sh   # ARPG-L checkpoint, tokenizer, reference batch, evaluator

torchrun --standalone --nnodes=1 --nproc_per_node=1 sample_c2i_ddp.py \
    --gpt-model ARPG-L \
    --gpt-ckpt weights/arpg_300m.pt \
    --vq-ckpt weights/vq_ds16_c2i.pt \
    --sample-schedule arccos --cfg-scale 5.0 --step 16 \
    --rejection-mode rejection \
    --confidence-metric random \
    --rejection-threshold 2.0 \
    --max-reject-rate 0.7 \
    --no-compile
```

`--max-reject-rate` is ρ. `--rejection-threshold 2.0` is above any score, so every token is a candidate and ρ alone decides how many are deferred. Leaving out `--rejection-mode` gives unmodified ARPG. The command writes 50,000 samples; add `--num-fid-samples 1000` for a quick check.

## What is in this repository

| Path | Contents |
|---|---|
| `models/arpg.py` | The ARPG model, with `generate_with_rejection` (RTR and the confidence-guided variants) and `generate_with_refinement` (a re-decoding ablation) added |
| `models/confidence.py` | The scoring rules: random, maximum probability, entropy and margin |
| `utils/rejection_tracker.py` | Per-step deferral logs and spatial heatmaps |
| `sample_c2i_ddp.py` | The sampling script, with the rejection flags added |
| `scripts/` | Asset download, baseline reproduction, sweeps, and parity tests that check the modified decoder reproduces ARPG's output exactly when nothing is deferred |
| `notebooks/` | The Colab notebooks that ran each experiment |
| `docs/` | Experiment logs, the baseline runbook and the progress report |

## Credits

The model, the training code and the pretrained checkpoints are from [ARPG](https://github.com/hp-l33/ARPG) by Haopeng Li, Jinyue Yang, Guoqi Li and Huan Wang, released under the MIT license, which this repository keeps in [LICENSE](LICENSE). Their instructions for training are in [GETTING_STARTED.md](GETTING_STARTED.md). If you use ARPG, cite their paper:

```bibtex
@inproceedings{li2026autoregressive,
    title={Autoregressive Image Generation with Randomized Parallel Decoding},
    author={Haopeng Li and Jinyue Yang and Guoqi Li and Huan Wang},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026}
}
```

Evaluation uses the [guided-diffusion](https://github.com/openai/guided-diffusion) evaluator, and the image tokenizer comes from [LlamaGen](https://github.com/FoundationVision/LlamaGen).
