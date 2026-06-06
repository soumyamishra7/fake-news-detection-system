import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading data...")
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

data["content"] = data["title"] + " " + data["text"]

X = data["content"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Converting text to numbers...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training the model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel accuracy: {accuracy * 100:.2f}%")
print("\nDetailed report:")
print(classification_report(y_test, predictions, target_names=["Fake", "Real"]))

print("\n--- Test your own news ---")
my_news = input("Paste a news headline or article text and press Enter:\n")
my_vec = vectorizer.transform([my_news])
result = model.predict(my_vec)[0]
confidence = model.predict_proba(my_vec)[0]

if result == 1:
    print(f"\nResult: REAL news (confidence: {confidence[1]*100:.1f}%)")
else:
    print(f"\nResult: FAKE news (confidence: {confidence[0]*100:.1f}%)")