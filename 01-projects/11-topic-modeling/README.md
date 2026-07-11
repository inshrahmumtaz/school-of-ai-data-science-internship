# Topic Modeling with NMF

## Business Problem

Automatically discover hidden themes in a large collection of news articles without using labels.

## Dataset

- BBC News Dataset
- 2,225 articles
- Categories include Business, Politics, Sport, Technology, and Entertainment.

## Preprocessing

- Lowercasing
- Removed punctuation
- Removed numbers
- Removed English stopwords
- Removed short words

## Model

- TF-IDF Vectorizer
- NMF (Non-negative Matrix Factorization)

Two configurations were tested:

- 8 Topics
- 12 Topics

The 8-topic model was selected because it produced more interpretable topics.

## Outputs

- topics.txt
- topic_examples.csv
- Topic word charts

## Limitations

- Topics are unlabeled and require human interpretation.
- Similar topics may overlap.
- Results depend on preprocessing and the chosen number of topics.

## Run

```bash
pip install -r requirements.txt

python src/train.py
python src/report.py
```