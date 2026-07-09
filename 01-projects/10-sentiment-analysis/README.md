# Sentiment Analysis with NLP

## Business Problem

Automatically classify customer reviews into positive or negative sentiment using Natural Language Processing (NLP). This helps organizations understand customer opinions at scale.

## Dataset

- IMDb Movie Reviews Dataset
- 50,000 labeled reviews
- Balanced positive and negative classes

## Text Preprocessing

- Lowercasing
- HTML tag removal
- Punctuation removal
- Stopword removal

## Models

- Baseline: CountVectorizer + Logistic Regression
- Improved: TF-IDF + Logistic Regression

## Evaluation Metrics

- Accuracy
- F1 Score
- ROC-AUC

## Key Visualizations

- Sentiment distribution
- Text length distribution
- Top positive words
- Top negative words
- Confusion matrix

## Error Analysis

Misclassifications were mainly caused by:
- Sarcasm
- Mixed opinions
- Very short reviews
- Ambiguous wording

## Run

```bash
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
```