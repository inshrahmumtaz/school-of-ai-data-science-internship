import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from data import load_data
from text_preprocess import clean_text

df = load_data()

df["clean_text"] = df["content"].apply(clean_text)

vectorizer = TfidfVectorizer(
    max_df=0.95,
    min_df=10
)

X = vectorizer.fit_transform(df["clean_text"])

model = NMF(
    n_components=8,
    random_state=42
)

model.fit(X)

joblib.dump(model, "outputs/nmf_model.joblib")
joblib.dump(vectorizer, "outputs/vectorizer.joblib")

print("Topic model saved successfully.")