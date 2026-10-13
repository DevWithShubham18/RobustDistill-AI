from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from robustdistill.core import MODELS, fgsm, get_data, make_model, pgd

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_NAMES = {
    "plain": "Weak student (standard)",
    "teacher": "Robust teacher",
    "robust_kd": "Robust-distilled student",
}
torch.set_num_threads(min(torch.get_num_threads(), 2))
st.set_page_config(page_title="RobustDistill AI Lab", page_icon="🧪", layout="wide")
st.markdown("""
<style>
.stApp{background:radial-gradient(ellipse at 10% 0%,#142a42 0,#0b1220 46%,#080d17 100%);color:#e7eef8}
[data-testid="stHeader"]{background:transparent}.hero{padding:1.25rem 1.5rem;border:1px solid #263c54;border-radius:18px;background:linear-gradient(115deg,#101e30,#102b36);margin-bottom:1.2rem}
.hero h1{margin:0;color:#f4f8ff;font-size:2.15rem}.hero p{color:#a9bbcf;margin:.45rem 0 0}
div[data-testid="stMetric"]{background:#111d2d;padding:.8rem;border:1px solid #26384c;border-radius:12px}
section[data-testid="stSidebar"]{background:#0c1523}
</style>
<div class="hero"><h1>🧪 RobustDistill AI Lab</h1><p>Adversarial robustness through knowledge distillation · 8×8 handwritten digits</p></div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    data = get_data(metrics["seed"])
    models = {}
    for name in ("teacher", *MODELS):
        model = make_model(name)
        model.load_state_dict(torch.load(ARTIFACTS / f"{name}.pt", map_location="cpu", weights_only=True))
        model.eval()
        models[name] = model
    return metrics, data, models


def input_image_tensor(image: Image.Image):
    """Resize grayscale uploads to model's 8x8 [0,1] input."""
    image = ImageOps.grayscale(image).resize((8, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    # Common uploads are black ink on white paper; invert to match sklearn digits.
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    return torch.from_numpy(arr.copy()).unsqueeze(0).unsqueeze(0).clamp(0, 1)


def probs(model, x):
    with torch.no_grad():
        return F.softmax(model(x), dim=1)[0]


metrics, data, models = load_artifacts()
with st.sidebar:
    st.subheader("Choose a handwritten digit")
    source = st.radio("Input source", ["Held-out test example", "Upload image"])
    truth = None
    if source == "Held-out test example":
        index = st.slider("Test-set image", 0, len(data.y_test)-1, 0)
        x = data.x_test[index:index+1]
        truth = int(data.y_test[index])
    else:
        upload = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
        if upload is None:
            st.info("Upload one handwritten digit to continue.")
            st.stop()
        uploaded_image = Image.open(upload)
        x = input_image_tensor(uploaded_image)
        if st.checkbox("I know the true digit"):
            truth = int(st.selectbox("True label", list(range(10))))
    st.divider()
    target_choice = st.selectbox("Attack the weak student using", ["True label (when known)", "Weak student's prediction"])
    attack_method = st.selectbox("Attack", ["PGD", "FGSM"])
    epsilon = st.slider("ε · L∞ pixel budget", 0.0, 0.35, 0.15, 0.01)
    steps = st.slider("PGD steps", 1, 30, 10) if attack_method == "PGD" else 1
    run = st.button("Generate Adversarial Attack", type="primary", use_container_width=True)

x = x.cpu()
clean_probs = {name: probs(model, x) for name, model in models.items()}
clean_display = x[0, 0].numpy()
st.subheader("1 · Original image and predictions")
left, right = st.columns([1, 3])
left.image(clean_display, clamp=True, width=220, caption=f"8×8 grayscale · true label: {truth if truth is not None else 'unknown'}")
cards = right.columns(3)
for col, name in zip(cards, ("plain", "teacher", "robust_kd")):
    p = clean_probs[name]
    digit, confidence = int(p.argmax()), float(p.max())
    correctness = "✓ correct" if truth is not None and digit == truth else ("✗ incorrect" if truth is not None else "")
    col.metric(MODEL_NAMES[name], f"{digit} {correctness}", f"{confidence:.1%} confidence")

if run:
    weak_pred = int(clean_probs["plain"].argmax())
    attack_label = truth if truth is not None and target_choice.startswith("True") else weak_pred
    y = torch.tensor([attack_label], dtype=torch.long)
    with torch.random.fork_rng():
        torch.manual_seed(1000 + (index if source == "Held-out test example" else 77))
        if attack_method == "FGSM":
            adv = fgsm(models["plain"], x, y, epsilon)
        else:
            adv = pgd(models["plain"], x, y, epsilon=epsilon, steps=steps, random_start=True)
    st.session_state["attack_result"] = {"adv": adv, "epsilon": epsilon, "steps": steps,
                                         "method": attack_method, "attack_label": attack_label}

st.subheader("2 · Real gradient-based adversarial attack")
result = st.session_state.get("attack_result")
if result is None:
    st.info("Choose FGSM or PGD and press **Generate Adversarial Attack**. The attack targets the original weak student.")
else:
    adv = result["adv"]
    adv_probs = {name: probs(model, adv) for name, model in models.items()}
    original = x[0, 0].numpy()
    attacked = adv[0, 0].numpy()
    delta = np.abs(attacked-original)
    a,b,c = st.columns(3)
    a.image(original, clamp=True, caption="Original", use_container_width=True)
    b.image(attacked, clamp=True, caption=f"{result['method']} · ε={result['epsilon']:.2f}", use_container_width=True)
    c.image(np.clip(delta*8,0,1), clamp=True, caption="Pixel difference ×8", use_container_width=True)
    st.caption(f"Actual L∞ change: {float(delta.max()):.4f}. The small 8×8 changes can be hard to see; all three models receive the exact same perturbed tensor.")
    pc = st.columns(3)
    for col, name in zip(pc, ("plain", "teacher", "robust_kd")):
        p=adv_probs[name]; digit=int(p.argmax()); conf=float(p.max())
        correctness = "✓ correct" if truth is not None and digit == truth else ("✗ incorrect" if truth is not None else "")
        col.metric(MODEL_NAMES[name], f"{digit} {correctness}", f"{conf:.1%} confidence")
    if truth is not None and int(adv_probs["plain"].argmax()) != truth and int(adv_probs["robust_kd"].argmax()) == truth:
        st.success("Verified on this input: the weak student fails while the robust-distilled student remains correct.")
    elif truth is not None:
        st.info("This input did not yield weak-fails/robust-student-succeeds. Predictions above show the actual outcome.")
    else:
        st.caption("No true label was supplied; attack labels use the weak student's clean prediction.")

st.subheader("3 · Measured held-out accuracy")
st.markdown("Actual stored results from the original archive. Dataset: **scikit-learn handwritten digits, 8×8**, stratified seed 2026 split; test set **360 images**. Attack: untargeted white-box L∞ PGD-10, ε=0.15, random start; the dataset-wide accuracy attack is run separately against each model.")
plot = {"Model": [], "Accuracy": [], "Evaluation": []}
for key,label in (("plain","Weak student"),("teacher","Robust teacher"),("robust_kd","Robust-distilled student")):
    item=metrics["models"][key]
    for kind,value in (("Clean",item["clean_accuracy"]),("PGD-10",item["pgd_robust_accuracy"])):
        plot["Model"].append(label); plot["Accuracy"].append(value); plot["Evaluation"].append(kind)
st.bar_chart(plot, x="Model", y="Accuracy", color="Evaluation")
cols=st.columns(3)
for col,key in zip(cols,("plain","teacher","robust_kd")):
    item=metrics["models"][key]
    col.metric(MODEL_NAMES[key], f"{item['clean_accuracy']:.1%} clean", f"{item['pgd_robust_accuracy']:.1%} PGD-10")
with st.expander("Research interpretation"):
    st.markdown("""
    - These are real, one-run exploratory measurements from the supplied archive, not results from the CVPR paper.
    - In this run the robust-distilled student reached 44.7% PGD accuracy, while the adversarial-training-only student reached 50.3%. The robust-KD variant therefore did **not** beat that control.
    - The paper is Huang et al., *Boosting Accuracy and Robustness of Student Models via Adaptive Adversarial Distillation*, IEEE/CVF CVPR 2023. This is a class-scale 8×8 proof of concept, not a reproduction of AdaAD or its CIFAR-10 experiments.
    - Interactive predictions are transfer outcomes from an attack against the weak student. Chart PGD accuracy uses a separate attack per model.
    """)
