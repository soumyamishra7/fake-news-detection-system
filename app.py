from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

print("Loading model...")
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)