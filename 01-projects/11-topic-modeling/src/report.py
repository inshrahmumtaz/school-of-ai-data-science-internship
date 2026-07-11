import joblib
import pandas as pd
import matplotlib.pyplot as plt

from data import load_data
from text_preprocess import clean_text

df = load_data()

df["clean_text"] = df["content"].apply(clean_text)

model = joblib.load("outputs/nmf_model.joblib")
vectorizer = joblib.load("outputs/vectorizer.joblib")

X = vectorizer.transform(df["clean_text"])

feature_names = vectorizer.get_feature_names_out()

# ---------- Topics.txt ----------

with open("outputs/topics.txt", "w", encoding="utf-8") as f:

    for topic_idx, topic in enumerate(model.components_):

        words = [
            feature_names[i]
            for i in topic.argsort()[-10:]
        ]

        f.write(
            f"Topic {topic_idx+1}: "
            + ", ".join(words)
            + "\n\n"
        )

# ---------- Topic Assignments ----------

topic_values = model.transform(X)

df["topic_id"] = topic_values.argmax(axis=1)

examples = []

for i in range(8):

    sample = df[df["topic_id"] == i].head(3)

    for _, row in sample.iterrows():

        examples.append({
            "topic_id": i,
            "example_text": row["title"]
        })

example_df = pd.DataFrame(examples)

example_df.to_csv(
    "outputs/topic_examples.csv",
    index=False
)

# ---------- Charts ----------

for topic_idx, topic in enumerate(model.components_):

    top = topic.argsort()[-10:]

    words = feature_names[top]

    weights = topic[top]

    plt.figure(figsize=(8,5))

    plt.barh(words, weights)

    plt.title(f"Topic {topic_idx+1}")

    plt.tight_layout()

    plt.savefig(
        f"reports/figures/topic_{topic_idx+1:02d}_words.png",
        dpi=200
    )

    plt.close()

print("Topic report generated.")