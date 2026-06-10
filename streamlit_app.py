import streamlit as st
import sys
import json
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

st.set_page_config(
    page_title="LogiCheck · Fallacy Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; font-size: 16px; font-weight: 300; }

:root {
    --primary:   #615fff;
    --surface:   #1d293d;
    --surface-2: #0f172b;
    --border:    #314158;
    --text-main: #e2e8f0;
    --text-muted:#94a3b8;
    --radius:    6px;
}

:root, [data-theme="light"], [data-theme="dark"] {
    --primary-color: var(--primary) !important;
    --background-color: var(--surface-2) !important;
    --secondary-background-color: var(--surface) !important;
    --text-color: var(--text-main) !important;
}

[data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"] {
    --st-color-background: var(--surface-2) !important;
    --st-color-secondary-background: var(--surface) !important;
    --st-color-text: var(--text-main) !important;
    --st-color-primary: var(--primary) !important;
}

body { background-color: var(--surface-2) !important; color: var(--text-main) !important; }
[data-testid="stAppViewContainer"] { background-color: var(--surface-2) !important; }

section[data-testid="stSidebar"] {
    background-color: var(--surface-2) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebarContent"] {
    display: flex;
    flex-direction: column;
}
section[data-testid="stSidebar"] * { color: var(--text-main) !important; }
.sidebar-info {
    display: flex; align-items: center;
    padding: 0px 8px; border-radius: 6px;
    font-size: 1rem; color: var(--text-main);
}

[data-testid="stHeader"] { height: 0px !important; background: transparent !important; }
.main .block-container { padding: 1rem 2rem !important; max-width: 1280px; background-color: var(--surface-2); }

.page-header { margin-bottom: 1.5rem; }
.page-header h1 {
    font-size: 2.8rem; font-weight: 400;
    color: var(--text-main); margin: 0;
    letter-spacing: -0.6px;
}

.section-heading {
    font-size: 1.15rem; font-weight: 400;
    color: var(--text-muted); text-transform: uppercase;
    letter-spacing: .06em; margin: 1.5rem 0 .75rem;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.3rem;
    transition: all 0.2s ease-in-out;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: var(--primary);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
}
.metric-card .label {
    font-size: .85rem; font-weight: 400;
    text-transform: uppercase; letter-spacing: .07em;
    color: var(--text-muted); margin-bottom: .3rem;
}
.metric-card .value {
    font-size: 1.85rem; font-weight: 400;
    color: var(--text-main); letter-spacing: -1px;
}
.metric-card .sub {
    font-size: .85rem; color: var(--text-muted); margin-top: .2rem;
}

.interp-box, .ok-box, .warn-box, .risk-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.4rem; margin-top: .75rem; margin-bottom: .75rem;
    font-size: 0.95rem; color: var(--text-main); line-height: 1.6;
}
.interp-box { border-left: 4px solid var(--primary); }
.ok-box { border-left: 4px solid #10b981; }
.warn-box { border-left: 4px solid #f59e0b; }
.risk-box { border-left: 4px solid #ef4444; }

.interp-box strong { color: var(--primary); }
.ok-box strong { color: #10b981; }
.warn-box strong { color: #f59e0b; }
.risk-box strong { color: #ef4444; }

.stNumberInput > div > div > input,
.stTextInput  > div > div > input,
.stSelectbox [data-baseweb="select"],
.stTextArea textarea {
    border-radius: 8px !important;
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-main) !important;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus, .stTextInput > div > div > input:focus {
    border-color: var(--primary) !important;
}
.stSelectbox > div > div > div { border: none !important; }

div.stButton > button[kind="primary"] {
    background: var(--primary) !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    transition: filter 0.2s;
}
div.stButton > button[kind="primary"]:hover { filter: brightness(1.1); }

div.stButton > button:not([kind="primary"]) {
    border-radius: 8px !important;
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-main) !important;
    transition: border-color 0.2s, color 0.2s;
}
div.stButton > button:not([kind="primary"]):hover {
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px; border-bottom: 2px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px 6px 0 0 !important;
    font-weight: 400 !important; font-size: 1rem !important;
    background-color: transparent !important;
    color: var(--text-muted) !important;
    padding: 10px 16px !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary) !important;
}

[data-testid="stTable"], [data-testid="stDataFrame"] {
    background-color: var(--surface) !important;
}
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv
load_dotenv()

from src.data import load_all_datasets, load_json
from src.utils import setup_logging

setup_logging("logs/")

DATA_DIR = PROJECT_ROOT / "data" / "raw"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "models"
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pt"

FALLACY_LABELS = [
    "appeal_to_authority",
    "appeal_to_majority",
    "appeal_to_nature",
    "appeal_to_tradition",
    "appeal_to_worse_problems",
    "false_dilemma",
    "hasty_generalization",
    "slippery_slope",
    "no_fallacy",
]

FALLACY_DESCRIPTIONS = {
    "appeal_to_authority": "Citing authority as proof without proper reasoning",
    "appeal_to_majority": "Treating popularity as evidence that a claim is true",
    "appeal_to_nature": "Assuming something is good because it is natural",
    "appeal_to_tradition": "Arguing something is correct because it has always been done",
    "appeal_to_worse_problems": "Dismissing an issue by pointing to a bigger one",
    "false_dilemma": "Presenting only two options when more exist",
    "hasty_generalization": "Broad conclusion from a small or unrepresentative sample",
    "slippery_slope": "Assuming a chain of events without justification",
    "no_fallacy": "Valid reasoning with no detected logical fallacy",
}

for key, val in {
    "last_result": None,
    "pipeline": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return None

def ensure_model_exists(config):
    if CHECKPOINT_PATH.exists():
        return str(CHECKPOINT_PATH)

    hf_repo = config.get("project", {}).get("hf_repo")
    hf_file = config.get("project", {}).get("hf_filename", "best_model.pt")

    if hf_repo and hf_repo != "your-username/logicheck-model":
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=hf_repo, filename=hf_file)
            return path
        except Exception as e:
            logging.getLogger(__name__).error(f"hf download failed: {e}")
    return None

@st.cache_data
def load_dataset_cached():
    try:
        frames = []
        for name in ["train", "dev", "test"]:
            p = DATA_DIR / f"{name}.json"
            if p.exists():
                df = load_json(str(p), name)
                df["split"] = name
                frames.append(df)
        if frames:
            return pd.concat(frames, ignore_index=True)
    except Exception as e:
        logging.getLogger(__name__).error(f"failed to load dataset: {e}")
    return None

def ibox(text, kind="info"):
    cls = {"info": "interp-box", "warn": "warn-box", "ok": "ok-box", "risk": "risk-box"}[kind]
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

def page_header(title: str, subtitle: str = ""):
    st.markdown(
        f'<div class="page-header">'
        f'<div><h1>{title}</h1>'
        + (f'<p style="margin:0;color:var(--text-muted);font-size:1.05rem">{subtitle}</p>' if subtitle else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )

def render_home():
    page_header("LogiCheck", "Context-aware multi-label logical fallacy detection")

    tab_overview, tab_data, tab_config, tab_eval = st.tabs([
        "Overview", "Data Exploration", "Model Config", "Performance"
    ])

    with tab_overview:
        st.markdown(
            "Welcome to **LogiCheck**, an advanced system designed to identify and explain logical fallacies in text. "
            "By combining transformer-based classification with generative AI explanations, LogiCheck helps users "
            "uncover flawed reasoning and improve critical thinking skills."
        )
        st.divider()
        df = load_dataset_cached()
        if df is not None:
            total = len(df)
            has_fallacy = df["labels"].apply(lambda x: any(l != "no_fallacy" for l in x) if x else False).sum()
            unique_labels = set()
            for labels in df["labels"]:
                unique_labels.update(labels)
            avg_labels = df["labels"].apply(len).mean()

        st.markdown('<div class="section-heading">Dataset Statistics</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, label, value, sub in [
            (c1, "Total Samples", f"{total:,}", "across all splits"),
            (c2, "Fallacious Texts", f"{has_fallacy:,}", f"{has_fallacy/total:.1%} of dataset"),
            (c3, "Unique Fallacies", f"{len(unique_labels)}", "label categories"),
            (c4, "Avg Labels", f"{avg_labels:.2f}", "multi-label density per sample"),
        ]:
            col.markdown(
                f'<div class="metric-card">'
                f'<div class="label">{label}</div>'
                f'<div class="value">{value}</div>'
                f'<div class="sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-heading">System Architecture</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        steps = [
            ("Stage 1: Detection", "DeBERTa-v3-small encoder with a compact multi-label head and Focal Loss."),
            ("Stage 2: Explanation", "Gemini 1.5 Flash generates educational explanations for detected fallacies."),
            ("Inference Pipeline", "Two-stage pipeline: classify text, then explain detected reasoning errors."),
        ]
        for col, (title, desc) in zip([c1, c2, c3], steps):
            col.markdown(
                f'<div class="metric-card" style="text-align:center">'
                f'<div style="font-weight:600;font-size:1.05rem;margin-bottom:.5rem;color:var(--text-main);">{title}</div>'
                f'<div class="sub" style="font-size:.85rem;line-height:1.5">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-heading">Core Fallacies</div>', unsafe_allow_html=True)
        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown(
                "**Ad Hominem**\n\nAttacking the person instead of the argument.\n\n"
                "**Slippery Slope**\n\nAsserting that one event will inevitably lead to a chain of others."
            )
        with fc2:
            st.markdown(
                "**False Dilemma**\n\nPresenting only two options when more exist.\n\n"
                "**Hasty Generalization**\n\nDrawing conclusions from insufficient evidence."
            )

    with tab_data:
        df = load_dataset_cached()
        if df is None:
            st.error("Dataset not found. Place train.json, dev.json, test.json in data/raw/.")
        else:
            view_option = st.selectbox("Select View:", ["Raw Data", "Distribution", "Splits"])

            if view_option == "Raw Data":
                st.caption(f"Showing all {len(df):,} records")
                display_df = df.copy()
                display_df["labels_str"] = display_df["labels"].apply(lambda x: ", ".join(x))
                st.dataframe(display_df[["text", "labels_str", "split"]].rename(columns={"labels_str": "labels"}), width="stretch", height=400)

            elif view_option == "Distribution":
                exploded = df.explode("labels")
                counts = exploded["labels"].value_counts().reset_index()
                counts.columns = ["Fallacy Type", "Count"]
                fig = px.bar(counts, x="Fallacy Type", y="Count", color="Count", color_continuous_scale=["#1d293d", "#615fff"])
                fig.update_layout(margin=dict(t=10, b=10), xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            elif view_option == "Splits":
                split_counts = df.groupby("split").size().reset_index(name="Count")
                fig_split = px.pie(split_counts, names="split", values="Count", color_discrete_sequence=["#615fff", "#94a3b8", "#e2e8f0"])
                fig_split.update_layout(margin=dict(t=10, b=10))
                st.plotly_chart(fig_split, use_container_width=True)

            st.markdown('<div class="section-heading">Sequence Analysis</div>', unsafe_allow_html=True)
            df["text_len"] = df["text"].apply(lambda x: len(x.split()))
            avg_len = df["text_len"].mean()
            max_len = df["text_len"].max()

            sc1, sc2 = st.columns(2)
            sc1.metric("Average Word Count", f"{avg_len:.1f}")
            sc2.metric("Maximum Word Count", f"{max_len}")

            fig_len = px.histogram(df, x="text_len", nbins=30, color_discrete_sequence=["#615fff"])
            fig_len.update_layout(margin=dict(t=10, b=10), xaxis_title="Word Count", yaxis_title="Frequency")
            st.plotly_chart(fig_len, use_container_width=True)

    with tab_config:
        config = load_config()
        if config:
            mc = config.get("model", {})
            tc = config.get("training", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Backbone", mc.get("backbone", "N/A").split("/")[-1])
            c2.metric("Epochs", tc.get("epochs", 5))
            c3.metric("Batch Size", tc.get("batch_size", 16))

            st.markdown('<div class="section-heading">Training Strategy</div>', unsafe_allow_html=True)
            tc1, tc2, tc3 = st.columns(3)
            tc1.markdown(f"**Learning Rate**: `{tc.get('learning_rate', 'N/A')}`")
            tc2.markdown(f"**Warmup Ratio**: `{tc.get('warmup_ratio', 'N/A')}`")
            tc3.markdown(f"**Weight Decay**: `{tc.get('weight_decay', 'N/A')}`")

            fc1, fc2, fc3 = st.columns(3)
            fc1.markdown(f"**FP16 Training**: `{'Enabled' if tc.get('fp16') else 'Disabled'}`")
            fc2.markdown(f"**Focal Gamma**: `{config.get('focal_loss', {}).get('gamma', 'N/A')}`")
            fc3.markdown(f"**Max Length**: `{config.get('data', {}).get('max_length', 'N/A')}`")

            st.markdown('<div class="section-heading">Label Weights</div>', unsafe_allow_html=True)
            lw = config.get("label_weights", {})
            if lw:
                lw_df = pd.DataFrame([{"Label": k.replace("_", " ").title(), "Weight": v} for k, v in lw.items()])
                fig_w = px.bar(lw_df, x="Label", y="Weight", color_discrete_sequence=["#615fff"])
                fig_w.update_layout(margin=dict(t=10, b=10), xaxis_tickangle=-45)
                st.plotly_chart(fig_w, use_container_width=True)

    with tab_eval:
        if not CHECKPOINT_PATH.exists():
            st.warning("No trained model found.")
        else:
            ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=True)
            metrics = ckpt.get("metrics", {})
            if metrics:
                mc = st.columns(4)
                for col, label, key in zip(mc, ["Macro F1", "Precision", "Recall", "Micro F1"], ["macro_f1", "macro_precision", "macro_recall", "micro_f1"]):
                    val = metrics.get(key, 0)
                    col.markdown(f'<div class="metric-card" style="text-align:center"><div class="label">{label}</div><div class="value" style="font-size:1.5rem">{val:.4f}</div></div>', unsafe_allow_html=True)

                per_class = metrics.get("per_class", {})
                if per_class:
                    st.markdown('<div class="section-heading">Per-Class F1</div>', unsafe_allow_html=True)
                    pc_df = pd.DataFrame([{"Fallacy": k.replace("_", " ").title(), "F1": v["f1"]} for k, v in per_class.items()])
                    fig_pc = px.bar(pc_df, x="Fallacy", y="F1", color="F1", color_continuous_scale=["#ef4444", "#22c55e"])
                    fig_pc.update_layout(margin=dict(t=10, b=10), xaxis_tickangle=-45)
                    st.plotly_chart(fig_pc, use_container_width=True)

                st.markdown('<div class="section-heading">Metric Definitions</div>', unsafe_allow_html=True)
                st.markdown(
                    "**Macro F1**: Unweighted mean of per-class F1 scores (treats all classes equally).\n\n"
                    "**Micro F1**: Aggregated globally across all labels (favors majority classes).\n\n"
                    "**Precision/Recall**: Balance between detection accuracy and completeness."
                )

def render_predict():
    # force reload to apply library and env fixes
    if "fixed_v4" not in st.session_state:
        st.session_state["pipeline"] = None
        st.session_state["fixed_v4"] = True

    page_header("Predict", "Analyse text for logical fallacies")

    if not CHECKPOINT_PATH.exists():
        st.warning("No trained model found. Train a model first to use prediction.")
        return

    with st.container(border=True):
        st.markdown('<div class="section-heading" style="margin-top: 0;">Input Workspace</div>', unsafe_allow_html=True)

        text = st.text_area(
            "Text to Analyse:",
            height=140,
            placeholder="e.g. You can't trust her climate research -- she's not even a physicist.",
        )

        context = st.text_input(
            "Context (Optional):",
            placeholder="e.g. The article argues that stricter rules improve fairness.",
            help="Providing context from the broader article or thread helps the model understand the nuance."
        )

        c1, c2 = st.columns([1, 5])
        with c1:
            submit_btn = st.button("Analyse for Fallacies", type="primary", use_container_width=True)

    if submit_btn:
        if not text.strip():
            st.warning("Please enter some text to analyse.")
            return

        try:
            if st.session_state["pipeline"] is None:
                with st.spinner("Loading model..."):
                    config = load_config()
                    model_path = ensure_model_exists(config)
                    if not model_path:
                        st.error("No model found. Please train a model or configure Hugging Face Hub in config.yaml.")
                        return

                    from src.pipeline import LogiCheckPipeline
                    st.session_state["pipeline"] = LogiCheckPipeline.from_config(
                        str(CONFIG_PATH), model_path
                    )

            pipeline = st.session_state["pipeline"]
            with st.spinner("Analysing reasoning structures..."):
                result = pipeline.predict(
                    text=text.strip(),
                    context=context.strip() if context.strip() else None,
                    explain=True,
                )

            st.session_state["last_result"] = result.to_dict()
            st.rerun()

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            ibox(
                "<strong>How to resolve:</strong><br>"
                "1. Make sure the model is trained (<code>python src/train.py</code>).<br>"
                "2. Confirm <code>models/best_model.pt</code> exists.<br>"
                "3. Or configure <code>hf_repo</code> in <code>config/config.yaml</code> to auto-download.<br>"
                "4. Check that all dependencies are installed.",
                kind="warn",
            )

    if st.session_state.get("last_result"):
        res = st.session_state["last_result"]

        st.divider()
        st.markdown('<div class="section-heading">Analysis Results</div>', unsafe_allow_html=True)

        has_fallacy = res.get("has_fallacy", False)
        detected = res.get("detected_fallacies", [])
        scores = res.get("confidence_scores", {})
        explanation = res.get("explanation", "")

        if has_fallacy:
            fallacy_names = ", ".join(f.replace("_", " ").title() for f in detected if f != "no_fallacy")
            ibox(
                f"<strong>FALLACY DETECTED</strong><br>"
                f"Detected Categories: <strong>{fallacy_names}</strong>",
                kind="risk",
            )
        else:
            ibox(
                "<strong>NO FALLACY DETECTED</strong><br>"
                "The reasoning appears valid. No logical fallacy was identified.",
                kind="ok",
            )

        if explanation:
            st.markdown("### Reasoning Explanation")
            ibox(explanation, kind="info")
            explainer_name = res.get("explainer_name", "Local Template")
            st.caption(f"Reasoning provided by: {explainer_name}")

        if scores:
            with st.expander("View Detailed Confidence Scores", expanded=True):
                score_df = pd.DataFrame([
                    {"Fallacy": k.replace("_", " ").title(), "Confidence": v}
                    for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)
                ])
                fig_scores = px.bar(
                    score_df, x="Confidence", y="Fallacy", orientation="h",
                    color="Confidence", color_continuous_scale=["#1d293d", "#615fff"],
                )
                fig_scores.update_layout(
                    margin=dict(t=20, b=10, l=10, r=10),
                    yaxis_title="",
                    showlegend=False,
                    height=300
                )
                st.plotly_chart(fig_scores, use_container_width=True)

        st.button("Clear Result", use_container_width=False)

pg = st.navigation([
    st.Page(render_home, title="Dashboard Overview", default=True),
    st.Page(render_predict, title="Predict & Analyse"),
])

pg.run()
