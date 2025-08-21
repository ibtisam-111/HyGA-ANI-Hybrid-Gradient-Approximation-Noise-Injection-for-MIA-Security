 HyGA-ANI: Hybrid Gradient Approximation with Adaptive Noise Injection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official implementation of "HyGA-ANI: Hybrid Gradient Approximation with Adaptive Noise Injection for Robust Membership Privacy" paper.

 Overview

HyGA-ANI is a novel defense framework against Membership Inference Attacks (MIAs) that provides strong privacy guarantees while maintaining model utility. Our method uses gradient-aware adaptive noise injection to protect against state-of-the-art attacks.

 Key Features

- Defense against 5 state-of-the-art MIAs (MrM, FeS-MIA, SMILE, IMIA, SeqMIA)
- Evaluation on 4 datasets (CIFAR-10, SVHN, Purchase-100, Texas-100)
- Adaptive noise scaling based on gradient sensitivity
- Theoretical privacy guarantees with bounded KL divergence
- Minimal utility loss (<2% accuracy drop)




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
│   ├── imia.py                    IMIA attack implementation
│   └── seqmia.py                  SeqMIA attack implementation
├── defenses/
│   ├── base_defense.py            Base defense class
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
│   └── run_ablation.py            Ablation study script
├── tests/
│   ├── test_models.py
│   ├── test_attacks.py
│   └── test_defenses.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_demo.ipynb
│   ├── 03_attack_demo.ipynb
│   └── 04_results_analysis.ipynb
├── requirements.txt               Python dependencies
├── environment.yml               Conda environment
├── Dockerfile                    
├── LICENSE
├── CITATION.cff                  Citation file
└── README.md                     Main documentation


