import streamlit as st
import torch
import pandas as pd
import plotly.express as px
import os

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

.big-title{
    text-align:center;
    font-size:50px;
    font-weight:bold;
    color:#00D4FF;
}

.sub-title{
    text-align:center;
    color:gray;
    font-size:18px;
}

.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# MODEL PATH
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "notebooks",
    "finbert_model"
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    return tokenizer, model


tokenizer, model = load_model()

model.eval()

# -------------------------------------------------
# LABEL MAP
# -------------------------------------------------

label_map = {
    0: "Bearish 📉",
    1: "Bullish 📈",
    2: "Neutral ➖"
}

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("📊 Project Overview")

st.sidebar.success("FinBERT Fine-Tuned")

st.sidebar.metric(
    "Accuracy",
    "87.23%"
)

st.sidebar.metric(
    "Macro F1",
    "82.45%"
)

st.sidebar.markdown("---")

st.sidebar.subheader("Models Built")

st.sidebar.write("""
✅ RNN

✅ LSTM

✅ GRU

✅ FinBERT
""")

st.sidebar.markdown("---")

st.sidebar.info(
    """
    Financial News Sentiment
    Prediction using Deep
    Learning and FinBERT
    """
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown(
    """
    <div class='big-title'>
    📈 Financial News Sentiment Analyzer
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='sub-title'>
    Powered by Fine-Tuned FinBERT
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------------------------
# KPI CARDS
# -------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Accuracy",
        "87.23%"
    )

with col2:
    st.metric(
        "Macro F1",
        "82.45%"
    )

with col3:
    st.metric(
        "Best Model",
        "FinBERT"
    )

st.markdown("---")

# -------------------------------------------------
# SAMPLE INPUTS
# -------------------------------------------------

st.subheader("📌 Try Sample Headlines")

c1, c2, c3 = st.columns(3)

sample_text = ""

with c1:
    if st.button("📉 Bearish Sample"):
        sample_text = (
            "JPMorgan cuts earnings forecast "
            "due to slowing demand."
        )

with c2:
    if st.button("📈 Bullish Sample"):
        sample_text = (
            "Tesla beats expectations and "
            "raises annual guidance."
        )

with c3:
    if st.button("➖ Neutral Sample"):
        sample_text = (
            "Microsoft will release quarterly "
            "results next week."
        )

# -------------------------------------------------
# INPUT
# -------------------------------------------------

text = st.text_area(
    "Enter Financial News or Tweet",
    value=sample_text,
    height=150
)

# -------------------------------------------------
# PREDICT
# -------------------------------------------------

if st.button("🚀 Predict Sentiment"):

    if text.strip() == "":

        st.warning(
            "Please enter a financial headline."
        )

    else:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():

            outputs = model(**inputs)

            probs = torch.softmax(
                outputs.logits,
                dim=1
            )

            prediction = torch.argmax(
                probs,
                dim=1
            ).item()

        st.markdown("---")

        # -------------------------------
        # PREDICTION CARD
        # -------------------------------

        if prediction == 0:

            st.error(
                f"Prediction: {label_map[prediction]}"
            )

        elif prediction == 1:

            st.success(
                f"Prediction: {label_map[prediction]}"
            )

        else:

            st.info(
                f"Prediction: {label_map[prediction]}"
            )

        # -------------------------------
        # PROBABILITIES
        # -------------------------------

        bearish = float(probs[0][0]) * 100
        bullish = float(probs[0][1]) * 100
        neutral = float(probs[0][2]) * 100

        probability_df = pd.DataFrame(
            {
                "Sentiment":[
                    "Bearish",
                    "Bullish",
                    "Neutral"
                ],
                "Probability":[
                    bearish,
                    bullish,
                    neutral
                ]
            }
        )

        st.subheader(
            "📊 Prediction Confidence"
        )

        fig = px.bar(
            probability_df,
            x="Sentiment",
            y="Probability",
            color="Sentiment",
            text="Probability"
        )

        fig.update_layout(
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            probability_df,
            use_container_width=True
        )

# -------------------------------------------------
# MODEL COMPARISON
# -------------------------------------------------

st.markdown("---")

st.subheader(
    "📈 Model Performance Comparison"
)

comparison_df = pd.DataFrame(
    {
        "Model":[
            "RNN",
            "GRU",
            "LSTM",
            "FinBERT"
        ],

        "F1 Score":[
            0.3832,
            0.6880,
            0.7073,
            0.8245
        ]
    }
)

fig = px.bar(
    comparison_df,
    x="Model",
    y="F1 Score",
    color="Model",
    text="F1 Score"
)

fig.update_layout(
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -------------------------------------------------
# PROJECT HIGHLIGHTS
# -------------------------------------------------

st.markdown("---")

st.subheader(
    "🏆 Project Highlights"
)

st.success(
    """
    ✅ Built Simple RNN

    ✅ Built LSTM

    ✅ Built GRU

    ✅ Fine-Tuned FinBERT

    ✅ Validation Accuracy: 87.23%

    ✅ Macro F1 Score: 82.45%

    ✅ Interactive Streamlit Dashboard
    """
)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")

st.caption(
    "Developed by Binduja | Financial News Sentiment Prediction using FinBERT"
)