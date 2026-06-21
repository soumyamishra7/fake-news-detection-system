import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

print("Loading data...")
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real], ignore_index=True)
data = data.dropna()
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data["content"] = data["title"] + " " + data["text"]

X = data["content"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Build and save a word -> coefficient lookup for keyword explainability
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]
word_scores = dict(zip(feature_names, coefficients))
joblib.dump(word_scores, "word_scores.pkl")

print("Model, vectorizer, and word scores saved successfully!")
print(f"Total vocabulary size: {len(feature_names)}")