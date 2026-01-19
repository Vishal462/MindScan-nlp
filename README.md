# MindScan: Hybrid Multi-Task NLP for Mental Health Assessment

MindScan is a comprehensive, multi-task Natural Language Processing (NLP) system designed to perform mental health classification, emotion recognition, and sentiment analysis from a single text input. Built with Streamlit, it provides a clean, interactive interface for real-time psychological and emotional assessment.

**Live Demo:** [Run MindScan here](https://mindscan-nlp.streamlit.app/)

---

## Key Features
- **Multi-Task Architecture:** Unifies three distinct diagnostic tasks—mental health status, granular emotion detection, and sentiment polarity—into a single pipeline.
- **Advanced Transformer Models:** Utilizes fine-tuned **RoBERTa** and **DistilBERT** architectures optimized for high contextual sensitivity.
- **Interactive Visualizations:** Features real-time probability distribution plots for each prediction to provide transparency and model interpretability.
- **Efficient Deployment:** Employs Parameter-Efficient Fine-Tuning (**PEFT**) with **LoRA** to maintain a lightweight footprint suitable for cloud deployment.

## Technical Methodology
MindScan leverages a hybrid, modular design where each task is handled by a dedicated, optimized model:

| Task | Base Model | Optimization Technique |
| :--- | :--- | :--- |
| **Mental Health Status** | **RoBERTa-base** | Custom-weighted cross-entropy loss to address class imbalance. |
| **Emotion Recognition** | **DistilBERT** | Parameter-Efficient Fine-Tuning (PEFT) using **LoRA** ($r=8$). |
| **Sentiment Analysis** | **DistilBERT** | Binary classification head for positive/negative polarity detection. |

### Datasets Used
- **Mental Health:** Sentiment Analysis for Mental Health dataset (Kaggle), covering categories like Depression, Suicidal, Anxiety, Stress, and Bipolar Disorder.
- **Emotions & Sentiment:** Sentiment and Emotion Analysis Dataset (Kaggle), encompassing Joy, Sadness, Anger, Fear, Love, and Surprise.

## Performance Results
The models demonstrate strong reliability across all evaluated tasks:

- **Mental Health Classifier:** Achieved **82.69% accuracy** and a macro F1-score of 0.799.
- **Emotion Detector:** Achieved **92.87% accuracy** with strong performance across high-frequency emotions like Sadness and Joy.
- **Sentiment Analyser:** Achieved **91.39% accuracy** and a **0.97 AUC** on the ROC curve.

## Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/mindscan-nlp.git](https://github.com/your-username/mindscan-nlp.git)
cd mindscan-nlp
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run locally
```bash
streamlit run project.py
```
## Research Paper
The code in this repository implements the concepts and methods described in my research work:
**Hybrid Multi-Task NLP for Comprehensive Mental Health Assessment" by Vishal Agarwal.**
For detailed explanations of the project’s design, methodology, and results, please refer to the full paper:  
[Download MindScan Research Paper](docs/MindScan_Reserach_Paper.pdf)


