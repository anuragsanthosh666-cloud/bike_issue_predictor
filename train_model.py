import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("bike_problems_1000.csv")

X_text = df["problem_description"]
y = df["problem_category"]

# FIT vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X_text)  # ✅ FIT HERE

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# SAVE AFTER FITTING
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "model.pkl")

print("✅ Model and vectorizer saved correctly")
import joblib

v = joblib.load("vectorizer.pkl")
m = joblib.load("model.pkl")

print("Vocabulary size:", len(v.vocabulary_))
print("Model classes:", m.classes_)
