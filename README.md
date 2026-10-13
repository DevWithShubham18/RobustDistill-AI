# RobustDistill AI Lab · 8×8 class demo

Fast Streamlit wrapper around the **original trained models and measured results** from `RobustDistill_AI_Lab_Class_Demo.zip`. No MNIST models are used and no training is needed.

## Run in GitHub Codespaces

Open the `RobustDistill-AI` repository in Codespaces. In the terminal:

```bash
python -m pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0
```

The existing archive checkpoints and metrics are committed under `artifacts/`. The built-in selector uses held-out sklearn digits. Uploads are converted to 8×8 grayscale and normalized to `[0,1]`.

## Deploy on Streamlit Community Cloud

Push this repository to GitHub, then at [share.streamlit.io](https://share.streamlit.io/) select repository `RobustDistill-AI`, the target branch, and `app.py`. The app uses CPU-only PyTorch dependencies. The public URL must be opened and checked after deployment before it can be called verified.

## Original experiment and results

Dataset: scikit-learn handwritten digits, 1,797 total 8×8 images; stratified seed-2026 split: 1,149 train, 288 validation, 360 test. Metrics are the stored one-run held-out results from the supplied ZIP. Dataset-wide robustness uses a separate untargeted white-box L∞ PGD-10 attack per model, ε=0.15, random start.

| Model | Clean accuracy | PGD-10 accuracy |
|---|---:|---:|
| Weak student (plain) | 91.94% | 10.00% |
| Robust teacher | 94.72% | 55.56% |
| Robust-distilled student | 90.28% | 44.72% |

The archive also reports 50.28% PGD accuracy for its adversarial-training-only student, above robust KD's 44.72% in this run. Results are exploratory, not an AdaAD reproduction. This is an 8×8 proof of concept, not the CVPR paper's CIFAR-10 experiments.

The interactive attack targets the weak student and is transferred to the teacher and robust-KD student. It is distinct from the chart's separately attacked per-model PGD metrics.
