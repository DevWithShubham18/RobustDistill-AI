# Class-demo experimental summary

> Real metrics measured by `train.py` on the sklearn handwritten-digits test split.
> This does **not** reproduce the IEEE/CVF CVPR 2023 AdaAD baseline.

**Data:** sklearn handwritten digits (8×8); training/validation/test split
**Attack:** white-box untargeted PGD-10, L∞ ε=0.15 with per-model attacks; randomized start
**Seed:** 2026

| Model | Clean accuracy | PGD robust accuracy | Params | Training seconds |
|---|---:|---:|---:|---:|
| teacher | 94.7% | 55.6% | 17,226 | 0.6 |
| plain | 91.9% | 10.0% | 4,546 | 0.2 |
| kd | 87.5% | 20.6% | 4,546 | 0.2 |
| robust_kd | 90.3% | 44.7% | 4,546 | 0.7 |
| adv_student | 92.5% | 50.3% | 4,546 | 0.5 |

## Interpretation

In this run, adversarial training only achieved 50.3% PGD robust accuracy, while robust KD achieved 44.7%. Robust KD did not beat this important control.

A matched adversarially trained (non-distilled) student is included as a control. Robust accuracy is calculated using white-box attacks against each model separately. Results are exploratory, not evidence of novelty or reproduction of published AdaAD numbers.

## Necessary next experiments

1. Reproduce official AdaAD baseline on CIFAR-10 or explicitly describe a scaled-down controlled approximation.
2. Evaluate FGSM, PGD with restarts, and AutoAttack at the same attack budget.
3. Test uncertainty weighting as an ablation and repeat over multiple seeds.
4. Report runtime, model size, attack parameters, and confidence intervals.
