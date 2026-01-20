from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

df = pd.read_csv("bike_complaints_560.csv")

X = df["complaint"]
y = df["label"]

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    stop_words="english",
    min_df=2
)

X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_vec, y)

joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "model.pkl")

print("✅ Model and vectorizer saved correctly")
