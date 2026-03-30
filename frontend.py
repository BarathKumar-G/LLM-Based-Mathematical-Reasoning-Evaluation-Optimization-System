import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# =========================
# CONFIG
# =========================
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
# Resolve adapter path relative to this project's training directory
ADAPTER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training", "finetuned_model")

device = "cuda" if torch.cuda.is_available() else "cpu"

from backend.backend import run_phase1, run_phase2, ask_model as base_ask_model
from training.trained_model import run_phase3


# =========================
# LOAD BASE MODEL
# =========================
def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()
    return model, tokenizer


# =========================
# LOAD FINETUNED MODEL
# =========================
def load_finetuned_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Loading LoRA adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # Verify LoRA loaded
    print(f"Model type: {type(model).__name__}")

    return model, tokenizer


# =========================
# CACHE MODELS (IMPORTANT)
# =========================
@st.cache_resource
def get_models():
    base_model, base_tokenizer = load_base_model()
    ft_model, ft_tokenizer = load_finetuned_model()

    dataset = load_dataset("gsm8k", "main")["test"]

    return base_model, base_tokenizer, ft_model, ft_tokenizer, dataset


# Load everything once
base_model, tokenizer, ft_model, ft_tokenizer, dataset = get_models()


# =========================
# UI
# =========================
st.set_page_config(page_title="LLM Math Evaluator", layout="wide")

st.title("🧠 LLM Math Reasoning Evaluation")
st.markdown("Compare performance of Phi-3-mini across **Baseline**, **SCoT (Prompt Engineering)**, and **LoRA Fine-tuned** phases.")


# =========================
# RUN BUTTON
# =========================
num_questions = st.slider("Number of test questions:", min_value=5, max_value=100, value=30, step=5)

if st.button("🚀 Run Evaluation"):
    with st.spinner("Running all 3 phases... This may take a few minutes."):

        acc1, res1 = run_phase1(base_model, tokenizer, device, dataset, limit=num_questions)
        acc2, res2 = run_phase2(base_model, tokenizer, device, dataset, limit=num_questions)

        # Phase 3 uses the fine-tuned model
        acc3, res3 = run_phase3(ft_model, ft_tokenizer, dataset, limit=num_questions)

        st.session_state["acc1"] = acc1
        st.session_state["acc2"] = acc2
        st.session_state["acc3"] = acc3

        st.session_state["res1"] = res1
        st.session_state["res2"] = res2
        st.session_state["res3"] = res3


# =========================
# ACCURACY COMPARISON
# =========================
if "acc1" in st.session_state:

    st.subheader("📊 Accuracy Comparison")

    df = pd.DataFrame({
        "Phase": ["Baseline", "SCoT", "Fine-tuned"],
        "Accuracy": [
            st.session_state["acc1"],
            st.session_state["acc2"],
            st.session_state["acc3"]
        ]
    })

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        colors = ["#e74c3c", "#f39c12", "#2ecc71"]  # Red → Yellow → Green
        bars = ax.bar(df["Phase"], df["Accuracy"], color=colors)
        ax.set_title("Accuracy by Phase")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.0)

        # Add percentage labels on bars
        for bar, acc in zip(bars, df["Accuracy"]):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')

        st.pyplot(fig)

    with col2:
        # Format accuracy as percentage for display
        display_df = df.copy()
        display_df["Accuracy"] = display_df["Accuracy"].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_df, use_container_width=True)

        # Show improvement metrics
        st.markdown("**Improvements:**")
        acc1 = st.session_state["acc1"]
        acc2 = st.session_state["acc2"]
        acc3 = st.session_state["acc3"]
        st.markdown(f"- SCoT vs Baseline: **+{(acc2-acc1)*100:.1f}%** points")
        st.markdown(f"- Fine-tuned vs SCoT: **+{(acc3-acc2)*100:.1f}%** points")
        st.markdown(f"- Fine-tuned vs Baseline: **+{(acc3-acc1)*100:.1f}%** points")


# =========================
# ERROR DISTRIBUTION
# =========================
def plot_errors(results, title):
    error_counts = {}

    for item in results:
        err = item["error_type"]
        error_counts[err] = error_counts.get(err, 0) + 1

    fig, ax = plt.subplots()
    colors_map = {"correct": "#2ecc71", "wrong_calculation": "#e74c3c", "no_answer": "#95a5a6"}
    colors = [colors_map.get(k, "#3498db") for k in error_counts.keys()]
    ax.pie(error_counts.values(), labels=error_counts.keys(), autopct="%1.1f%%", colors=colors)
    ax.set_title(title)

    return fig


if "res1" in st.session_state:

    st.subheader("📉 Error Distribution")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.pyplot(plot_errors(st.session_state["res1"], "Baseline Errors"))

    with col2:
        st.pyplot(plot_errors(st.session_state["res2"], "SCoT Errors"))

    with col3:
        st.pyplot(plot_errors(st.session_state["res3"], "Fine-tuned Errors"))


# =========================
# MODEL OUTPUTS
# =========================
if "res1" in st.session_state:

    st.subheader("🧠 Model Predictions")

    tab1, tab2, tab3 = st.tabs(["Phase 1 (Baseline)", "Phase 2 (SCoT)", "Phase 3 (Fine-tuned)"])

    def display_results(results):
        for item in results:
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"Prediction: `{item['prediction']}`")
            st.markdown(f"Actual: `{item['actual']}`")

            if item["correct"]:
                st.success("Correct ✅")
            else:
                st.error(f"Wrong ❌ — {item['error_type']}")

            st.divider()

    with tab1:
        display_results(st.session_state["res1"])

    with tab2:
        display_results(st.session_state["res2"])

    with tab3:
        display_results(st.session_state["res3"])


# =========================
# CUSTOM INPUT
# =========================
st.subheader("✏️ Try Your Own Question")

user_q = st.text_input("Enter a math question:")

if st.button("Solve"):
    if user_q:
        prompt = (
            "You are a highly accurate mathematician. Solve the following math problem.\n\n"
            "Instructions:\n"
            "1. Break the problem into clear steps.\n"
            "2. Perform each calculation carefully.\n"
            "3. Write your final answer as: #### <number>\n\n"
            f"Question: {user_q}"
        )

        output = base_ask_model(prompt, base_model, tokenizer, device)

        st.write("### Answer:")
        st.code(output)