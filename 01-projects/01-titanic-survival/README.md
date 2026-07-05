# Titanic Survival Prediction (Classification)

## Project Overview

This project predicts whether a passenger survived the Titanic disaster using machine learning. The workflow includes data exploration, preprocessing, model training, evaluation, and saving the trained model.

---

## Dataset

- **Dataset:** Titanic Survival Dataset
- **Source:** Kaggle Titanic Dataset
- **Target Variable:** `Survived`
- **Features Used:**
  - Passenger Class (Pclass)
  - Sex
  - Age
  - SibSp
  - Parch
  - Fare
  - Embarked
  - and other passenger information

---

## Project Structure

```
01-titanic-survival
│
├── data
│   ├── raw
│   └── processed
│
├── notebooks
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
│
├── src
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
│
├── reports
│   └── figures
│
├── models
│
└── README.md
```

---

## Exploratory Data Analysis (EDA)

The following analyses were performed:

- Dataset inspection
- Missing value analysis
- Survival distribution
- Survival by Gender
- Survival by Passenger Class
- Age distribution by Survival

### Key Insights

- Female passengers had a much higher survival rate.
- First-class passengers survived more often than lower classes.
- Age contained missing values that required imputation.
- Passenger class strongly influenced survival.
- Sex was one of the most important predictors.

---

## Data Preprocessing

The preprocessing pipeline includes:

- Median imputation for numerical features
- Most frequent imputation for categorical features
- One-Hot Encoding for categorical variables
- Train-Test Split (80/20)

---

## Models

Two machine learning models were trained:

- Logistic Regression
- Random Forest Classifier

Random Forest produced the best overall performance.

---

## Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | 97.98% |
| F1 Score | 97.33% |
| ROC-AUC | 99.73% |

---

## Visualizations

Generated figures include:

- Survival by Sex
- Survival by Passenger Class
- Age Distribution
- Confusion Matrix

---

## How to Run

Create a virtual environment:

```bash
python -m venv .venv
```

Activate environment (Windows):

```bash
.venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train.py
```

Evaluate the model:

```bash
python src/evaluate.py
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## What I Learned

- Performing exploratory data analysis
- Handling missing values
- Building preprocessing pipelines
- Training classification models
- Evaluating machine learning models
- Saving trained models for reuse
- Organizing a machine learning project using a professional folder structure

---

## Author

**Inshra Mumtaz**

School of AI – Data Science Internship