import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

print("Loading data...")

# Use ONLY the neutral WELFake dataset
data = pd.read_csv("WELFake_Dataset.csv")

print("Columns:", data.columns.tolist())
print("Sample:\n", data.head(2))

# Clean the data
data = data.dropna()
data["content"] = data["title"] + " " + data["text"]
data = data[["content", "label"]]

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