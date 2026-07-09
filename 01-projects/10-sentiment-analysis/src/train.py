import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data import load_data
from text_preprocess import clean_text

X_train, X_test, y_train, y_test = load_data()

X_train = X_train.apply(clean_text)
X_test = X_test.apply(clean_text)

pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

joblib.dump(pipeline, "models/model.joblib")

print("Model saved.")