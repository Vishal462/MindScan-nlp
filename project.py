import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

st.set_page_config(
    page_title="MindScan",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap');
:root {
    --primary-color: #4A4A4A; 
    --background-color: #FFFFFF; 
    --secondary-background-color: #F0F0F0; 
    --text-color: #111111;
    --risk-color: #DD0000; 
}
body, .stApp {
    font-family: 'Inter', sans-serif;
    color: var(--text-color);
    background-color: var(--background-color);
}
[data-testid*="stMetricDelta"], [data-testid="stAlert"] svg, 
.st-emotion-cache-1c5c3p0, .st-emotion-cache-1gh727j, .st-emotion-cache-pkbazv {
    color: inherit !important;
    display: none !important; 
}
.st-emotion-cache-1gh727j {
    color: #111111 !important;
}
.stTextArea textarea {
    transition: box-shadow 0.2s ease-in-out, border-color 0.2s ease-in-out;
    border: 1px solid #999999;
}
.stTextArea textarea:focus {
    border-color: #111111;
    box-shadow: 0 0 0 1px #111111;
}
.stButton button {
    background-color: #333333;
    color: white;
    padding: 10px 20px;
    border: 1px solid #111111;
    transition: background-color 0.3s ease, transform 0.1s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    font-weight: 600;
}
.stButton button:hover {
    background-color: #111111;
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
}
.stButton button:active {
    transform: translateY(0);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.card {
    border: 1px solid #111111; 
    border-radius: 8px;
    padding: 15px; 
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    height: 100%; 
    background-color: #FAFAFA; 
}
.card h4 {
    border-bottom: 1px solid #DDDDDD;
    padding-bottom: 8px;
    margin-bottom: 10px;
    font-weight: 700;
}
[data-testid="stMetric"] {
    background-color: #EFEFEF;
    border: 1px solid #DDDDDD;
    border-radius: 8px;
    padding: 20px 15px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}
[data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 2.5rem !important;
    color: #111111 !important;
    font-weight: 900 !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    color: #555555 !important;
}
.stDataFrame {
    padding: 0;
    margin: 0;
}
.high-risk-alert {
    background-color: #FFF0F0; 
    border: 2px solid var(--risk-color);
    padding: 15px;
    border-radius: 4px;
    color: #111111;
    font-weight: 600;
}
.low-risk-alert {
    background-color: #FAFAFA;
    border: 1px solid #EAEAEA;
    padding: 15px;
    border-radius: 4px;
    color: #555;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

CONDITION_LABELS = {0: 'Anxiety', 1: 'Bipolar', 2: 'Depression', 3: 'Normal', 4: 'Personality disorder', 5: 'Stress', 6: 'Suicidal'}
SENTIMENT_LABELS = {0: "Negative", 1: "Positive"}
EMOTION_LABELS = {0:'Sadness', 1:'Joy',2:'Love',3:'Anger',4:'Fear',5:'Surprise'}

@st.cache_resource
def load_model_checkpoint(model_path, task_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels
           )
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_lora_model(lora_path, base_id_or_path, task_name, num_labels):
    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_id_or_path,
            num_labels=num_labels,
        )
        lora_model = PeftModel.from_pretrained(base_model, lora_path)
        model = lora_model.merge_and_unload()
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"🚨 {task_name} (LoRA) Error: Ensure 'peft' is installed and base model ID is correct. Error: {e}",
                 icon=None)
        return None, None

# M1: MENTAL HEALTH CLASSIFICATION (RoBERTa - Standard Checkpoint)
CONDITION_TOKENIZER, CONDITION_MODEL = load_model_checkpoint(
    model_path='./mental_health_status_roberta_model_assets',
    task_name="MENTAL HEALTH (RoBERTa)",
    num_labels=len(CONDITION_LABELS)
)

# M2: SENTIMENT (DistilBERT - Standard Checkpoint)
SENTIMENT_TOKENIZER, SENTIMENT_MODEL = load_model_checkpoint(
    model_path='./sentiment_model_assets',
    task_name="SENTIMENT (DistilBERT)",
    num_labels=len(SENTIMENT_LABELS)
)

# M3: EMOTION (DistilBERT + LoRA Adapter)
EMOTION_TOKENIZER, EMOTION_MODEL = load_lora_model(
    lora_path='./emotions_model_assets',
    base_id_or_path='bhadresh-savani/distilbert-base-uncased-emotion',
    task_name="EMOTION (DistilBERT LoRA)",
    num_labels=len(EMOTION_LABELS)
)

def predict_model(text, tokenizer, model, labels):
    if model is None or tokenizer is None:
        return "MODEL ERROR", 0.0, np.array([])
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = F.softmax(outputs.logits, dim=1)[0]
    predicted_index = torch.argmax(probabilities).item()
    confidence_score = probabilities[predicted_index].item()
    predicted_label = labels.get(predicted_index, "Unknown")
    return predicted_label, confidence_score, probabilities.numpy()

def display_counts(text):
    char_count = len(text)
    word_count = len(text.split()) if text.strip() else 0
    st.markdown(
        f'<div style="text-align: right; color: #555555; font-size: 0.8em; margin-top: -10px;">CHARACTERS: {char_count} | WORDS: {word_count}</div>',
        unsafe_allow_html=True)

st.title("MINDSCAN NLP ANALYZER: MULTI-MODEL INTEGRATION")
st.caption("Integrated Analysis for Mental Health, Sentiment, and Granular Emotion")
st.markdown("---")

user_text = st.text_area(
    "Enter the text for analysis:",
    value="",
    height=250,
)

display_counts(user_text)
analyze_button = st.button("RUN MULTI-MODEL ANALYSIS", use_container_width=True, key="main_button_analysis")
if analyze_button and user_text:
    with st.spinner('Analyzing text with three integrated models...'):
        text_to_analyze = user_text
        if not text_to_analyze:
            st.warning("Please enter text to run the analysis.", icon="📝")
            st.stop()
        if CONDITION_MODEL is None and SENTIMENT_MODEL is None and EMOTION_MODEL is None:
            st.error("Cannot run analysis: All models failed to load.", icon=None)
            st.stop()
        st.markdown("---")
        st.subheader("ANALYSIS RESULTS")

        # Split display into three columns for side-by-side primary results
        col_condition, col_sentiment, col_emotion = st.columns(3)

        high_risk_flag = False
        top_condition = "N/A"
        sentiment = "N/A"
        top_emotion = "N/A"

        # --- MODEL 1: MENTAL HEALTH CLASSIFICATION (RoBERTa) ---
        with col_condition:
            st.markdown("##### 1. MENTAL HEALTH")
            if CONDITION_MODEL is not None:
                condition, condition_confidence, condition_probabilities = predict_model(
                    text_to_analyze, CONDITION_TOKENIZER, CONDITION_MODEL, CONDITION_LABELS
                )
                top_condition = condition

                if condition in ['Suicidal', 'Depression', 'Anxiety', 'Bipolar']:
                    high_risk_flag = True
                st.metric(
                    label="Primary Status",
                    value=condition.upper(),
                    delta=f"Confidence: {condition_confidence:.2f}"
                )
                # Store data for visualization
                st.session_state['prob_df_condition'] = pd.DataFrame({
                    'Condition': list(CONDITION_LABELS.values()),
                    'Probability': condition_probabilities
                }).sort_values(by='Probability', ascending=False)
            else:
                st.warning("M1 Model not available.", icon=None)

        # --- MODEL 2: SENTIMENT ANALYSIS (DistilBERT) ---
        with col_sentiment:
            st.markdown("##### 2. DISCOURSE SENTIMENT")
            if SENTIMENT_MODEL is not None:
                sentiment, sentiment_confidence, sentiment_probabilities = predict_model(
                    text_to_analyze, SENTIMENT_TOKENIZER, SENTIMENT_MODEL, SENTIMENT_LABELS
                )
                st.metric(
                    label="Overall Polarity",
                    value=sentiment.upper(),
                    delta=f"Confidence: {sentiment_confidence:.2f}"
                )
                # Store data for visualization
                st.session_state['prob_df_sentiment'] = pd.DataFrame({
                    'Polarity': list(SENTIMENT_LABELS.values()),
                    'Probability': sentiment_probabilities
                }).sort_values(by='Probability', ascending=False)
            else:
                st.warning("M2 Model not available.", icon=None)

        # --- MODEL 3: EMOTION DETECTION (DistilBERT LoRA) ---
        with col_emotion:
            st.markdown("##### 3. GRANULAR EMOTION")
            if EMOTION_MODEL is not None:
                top_emotion, emotion_confidence, emotion_probabilities = predict_model(
                    text_to_analyze, EMOTION_TOKENIZER, EMOTION_MODEL, EMOTION_LABELS
                )
                top_emotion = top_emotion
                st.metric(
                    label="Dominant Emotion",
                    value=top_emotion.upper(),
                    delta=f"Confidence: {emotion_confidence:.2f}"
                )
                # Store data for visualization
                st.session_state['prob_df_emotion'] = pd.DataFrame({
                    'Emotion': list(EMOTION_LABELS.values()),
                    'Probability': emotion_probabilities
                }).sort_values(by='Probability', ascending=False)
            else:
                st.warning("M3 Model not available.", icon=None)

        st.markdown("---")

        with st.expander("VIEW DETAILED CONFIDENCE DISTRIBUTIONS", expanded=False):

            st.header("CONFIDENCE DISTRIBUTION ")
            col_dist_condition, col_dist_sentiment, col_dist_emotion = st.columns(3)

            def display_confidence_dataframe(df_key, title, category_name, col_container):
                with col_container:
                    st.markdown(
                        f"""
                        <div class="card" style="padding: 15px;">
                        <h4 style="margin-bottom: 5px;">{title}</h4>
                        """, unsafe_allow_html=True
                    )
                    if df_key in st.session_state:
                        df_display = st.session_state[df_key].copy()
                        df_display.columns = [category_name, 'Confidence']

                        st.dataframe(
                            df_display,
                            column_config={
                                "Confidence": st.column_config.ProgressColumn(
                                    "Confidence",
                                    format="%.2f",
                                    min_value=0,
                                    max_value=1.0,
                                ),
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)

            display_confidence_dataframe('prob_df_condition', 'Mental Health Classification', 'Condition',
                                         col_dist_condition)

            # Sentiment Distribution
            display_confidence_dataframe('prob_df_sentiment', 'Discourse Sentiment Analysis', 'Polarity',
                                         col_dist_sentiment)

            # Emotion Distribution
            display_confidence_dataframe('prob_df_emotion', 'Granular Emotion Detection', 'Emotion', col_dist_emotion)

        st.markdown("---")
        st.header("INTERPRETATION & RISK ASSESSMENT")

        if high_risk_flag:
            st.markdown(
                f"<div class='high-risk-alert'><strong>HIGH CONCERN ASSESSMENT:</strong> Primary status is <strong>{top_condition.upper()}</strong>. This classification, combined with a <strong>{sentiment.upper()}</strong> sentiment and <strong>{top_emotion.upper()}</strong> emotion, suggests significant distress.</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='low-risk-alert'><strong>LOW-TO-MODERATE ASSESSMENT:</strong> Primary status is <strong>{top_condition.upper()}</strong>. The overall profile suggests situational stress or normal emotional responses.</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #555555; font-weight: 600; margin-top: 20px; padding-top: 10px; border-top: 1px dashed #DDDDDD;'>
            This tool is for information only and does not constitute medical advice.
            </div>
            """, unsafe_allow_html=True
        )
