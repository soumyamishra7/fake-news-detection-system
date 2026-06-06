from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

print("Loading and training model...")
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

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
print("Model ready!")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    news_input = ""
    if request.method == "POST":
        news_input = request.form["news"]
        vec = vectorizer.transform([news_input])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        if prediction == 1:
            result = "REAL"
            confidence = round(proba[1] * 100, 1)
        else:
            result = "FAKE"
            confidence = round(proba[0] * 100, 1)
    return render_template("index.html", result=result, confidence=confidence, news_input=news_input)

if __name__ == "__main__":
    app.run(debug=True)