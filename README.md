# AI-Powered Sentiment Analysis and Issue Classification System

An end-to-end NLP system that performs sentiment classification on customer reviews and automatically categorizes negative reviews into specific issue types for actionable customer feedback analysis. The project also includes an interactive Streamlit analytics dashboard to visualize sentiment distribution and issue breakdowns.

---

## System Overview

The system operates in a two-stage NLP pipeline:

1. Stage 1 — Sentiment Analysis:
   - Classifies customer reviews into Positive or Negative sentiment.
   - Evaluated with both a lightweight baseline model (TF-IDF + Logistic Regression) and a fine-tuned deep transformer model (RoBERTa-base).

2. Stage 2 — Negative Issue Classification:
   - Filters reviews flagged as Negative to determine root causes.
   - Categorizes complaints into granular issue classes using rule-based/spaCy keyword matching, zero-shot classification, and a fine-tuned BART (facebook/bart-large-mnli) sequence classifier.

3. Analytics Dashboard:
   - Built with Streamlit and Plotly to visualize overall sentiment breakdown and granular complaint categories from uploaded CSV data.

---

## Dataset and Preprocessing

- Data Source: Amazon Product Reviews (arhamrumi/amazon-product-reviews via KaggleHub / Reviews.csv).
- Target Field: Review Text.
- Initial Sentiment Annotation: Weak supervision via distilbert-base-uncased-finetuned-sst-2-english (with input truncation to 512 characters).
  - Positive reviews: 389,055
  - Negative reviews: 179,399
- Data Balancing and Sampling:
  - Balanced subset created with equal class distribution (179,399 per class).
  - Sampled to a 50,000-sample balanced dataset (25,000 Positive / 25,000 Negative) for fine-tuning.
  - Stratified 80/20 train-validation split (40,000 training, 10,000 validation).

---

## Models and Performance

### 1. Sentiment Analysis Models

| Model | Architecture | Hyperparameters / Details | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Lightweight Baseline | TF-IDF + Logistic Regression | Stratified 80/20 split, saved with Joblib (logistic_model.pkl, tfidf_vectorizer.pkl) | 85.00% | 0.85 | 0.85 | 0.85 |
| Deep Transformer | roberta-base | Sequence length: 256, Batch size: 16 (train) / 32 (eval), Gradient accumulation: 2, Learning rate: 2e-5, Weight decay: 0.01, Warmup: 500, Epochs: 3, FP16 enabled, Early stopping (patience: 3) | 91.96% | 89.99% | 94.42% | 92.15% |

### 2. Issue Classification Models (Negative Reviews)

Negative reviews are classified into specific issue categories:
- delivery issue
- product quality issue
- service issue
- pricing issue
- return/refund issue
- website/app issue
- payment issue
- promotions/discount issue
- recommendation/feedback
- other

Implemented Approaches:
- spaCy Lemmatized Keyword Matching: Rule-based categorization using en_core_web_sm and category-specific lemma dictionaries.
- Zero-Shot Classification: Using facebook/bart-large-mnli pipeline for multi-label / single-label candidate matching.
- Fine-Tuned Multi-Class Classifier:
  - Backbone: facebook/bart-large-mnli (BartForSequenceClassification with 8 output classes).
  - Training config: Batch size 4/8, learning rate 2e-5, weight decay 0.01, FP16, early stopping.
  - Results:
    - Accuracy: 87.10%
    - Precision: 75.86%
    - Recall: 87.10%
    - F1-Score: 81.09%
    - Loss: 0.8618

---

## Interactive Dashboard (app.py)

A Streamlit application designed for customer response analysis:
- File Uploader: Upload review CSV files containing response_type (Positive/Negative) and category fields.
- Overall Sentiment Chart: Horizontal bar chart comparing Positive vs. Negative response percentages.
- Negative Category Breakdown: Interactive bar chart displaying the percentage distribution of complaints across issue categories.

---

## Repository Structure

- AI_Powered_Sentiment_Analysis_&_Issue_Classification_System.ipynb: Complete training and evaluation notebook
- README.md: Project documentation

---

## Prerequisites and Installation

To run the notebook or application, install the required dependencies:

pip install torch transformers datasets scikit-learn pandas numpy matplotlib spacy streamlit plotly joblib kagglehub

python -m spacy download en_core_web_sm

---

## How to Run

### 1. Training and Evaluation Notebook
Open and run AI_Powered_Sentiment_Analysis_&_Issue_Classification_System.ipynb in Google Colab or Jupyter Notebook:
- Run the dataset download and preprocessing cells.
- Train or test the TF-IDF + Logistic Regression model.
- Fine-tune or run inference with roberta-base for sentiment analysis.
- Run the negative feedback categorization and fine-tune facebook/bart-large-mnli.

### 2. Running the Streamlit Dashboard
Generate app.py from the notebook or save the dashboard script, then run:

streamlit run app.py

Upload a CSV file containing response_type and category columns to view the analysis report.
