HyGA-ANI: Hybrid Gradient Approximation Noise Injection for Robust Membership Privacy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

Official implementation of "HyGA-ANI: Hybrid Gradient Approximation Noise Injection for Robust Membership Privacy".

 Overview

HyGA-ANI is a novel defense framework against Membership Inference Attacks (MIAs) that uses gradient-aware adaptive noise injection to provide strong privacy guarantees while maintaining model utility.

Key features:
- Gradient-aware vulnerability scoring
- Adaptive noise injection mechanism
- Theoretical privacy guarantees ((ε, δ)-differential privacy)
- Support for both image and tabular data
- Defense against state-of-the-art attacks (MrM, FeS-MIA, SMILE, IMIA, SeqMIA)

## Results

Our method achieves:
- MIA accuracy reduction to 52.1-55.3% on image data (near-random guessing)
- MIA accuracy reduction to 59.6-62.4% on tabular data
- Less than 2% model utility loss
- 1.8-2.7× computational overhead
Installation

```bash
 Clone the repository
git clone https://github.com/yourusername/HyGA-ANI.git
cd HyGA-ANI

 Install dependencies
pip install -r requirements.txt

HyGA-ANI/
├── .github/
│   └── workflows/
│       └── ci.yml                 CI/CD configuration
├── configs/
│   ├── base.yaml                  Base configuration
│   ├── cifar10.yaml               CIFAR-10 specific config
│   ├── svhn.yaml                  SVHN specific config
│   ├── purchase100.yaml           Purchase-100 specific config
│   └── texas100.yaml              Texas-100 specific config
├── data/
│   ├── preprocess.py              Data preprocessing script
│   └── README.md                  Data documentation
├── models/
│   ├── resnet.py                  ResNet implementation
│   ├── tabnet.py                  TabNet implementation
│   └── hyga_ani.py                HyGA-ANI defense implementation
├── attacks/
│   ├── mrm.py                     MrM attack implementation
│   ├── fes_mia.py                 FeS-MIA attack implementation
│   ├── smile.py                   SMILE attack implementation
│   ├── imia.py                    # IMIA attack implementation
│   └── seqmia.py                  # SeqMIA attack implementation
├── defenses/
│   ├── base_defense.py            # Base defense class
│   ├── dynanoise.py               DynaNoise implementation
│   ├── noise_transition.py        Noise Transition implementation
│   ├── adani.py                   AdaNI implementation
│   ├── adamixup.py                AdaMixup implementation
│   └── hyga_ani_defense.py        HyGA-ANI defense implementation
├── evaluation/
│   ├── metrics.py                 Evaluation metrics
│   ├── visualize.py               Visualization utilities
│   └── results/                   Precomputed results
│       ├── cifar10/
│       ├── svhn/
│       ├── purchase100/
│       └── texas100/
├── scripts/
│   ├── train.py                   Training script
│   ├── evaluate.py                Evaluation script
│   ├── run_attacks.py             Attack execution script
│   └── run_ablation.py            blation study script
├── tests/
│   ├── test_models.py
│   ├── test_attacks.py
│   └── test_defenses.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_demo.ipynb
│   ├── 03_attack_demo.ipynb
│   └── 04_results_analysis.ipynb
└── README.md                     # Main documentation

