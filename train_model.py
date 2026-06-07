import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

print("Loading data...")

# Original dataset
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")
fake["label"] = 0
real["label"] = 1
data1 = pd.concat([fake, real], ignore_index=True)
data1["content"] = data1["title"] + " " + data1["text"]
data1 = data1[["content", "label"]]

# New dataset
data2 = pd.read_csv("fake_or_real_news.csv")
data2["label"] = data2["label"].map({"REAL": 1, "FAKE": 0})
data2["content"] = data2["title"] + " " + data2["text"]
data2 = data2[["content", "label"]]

# Combine both datasets
data = pd.concat([data1, data2], ignore_index=True)
data = data.dropna()
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total articles: {len(data)}")
print(f"Fake: {len(data[data.label==0])} | Real: {len(data[data.label==1])}")

X = data["content"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Model saved successfully!")