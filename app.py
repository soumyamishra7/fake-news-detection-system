from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

print("Loading model...")
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
word_scores = joblib.load("word_scores.pkl")
print("Model ready!")

def get_top_keywords(text, vec, top_n=5):
    """Find which words in the input most influenced the prediction."""
    feature_array = vec.toarray()[0]
    word_list = vectorizer.get_feature_names_out()
    
    present_words = []
    for idx, val in enumerate(feature_array):
        if val > 0:
            word = word_list[idx]
            score = word_scores.get(word, 0)
            present_words.append((word, score, val))
    
    # Sort by absolute influence (score * tfidf weight)
    present_words.sort(key=lambda x: abs(x[1] * x[2]), reverse=True)
    return present_words[:top_n]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    news_input = ""
    keywords = []
    is_uncertain = False

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

        if confidence < 60:
            is_uncertain = True

        top_words = get_top_keywords(news_input, vec, top_n=5)
        for word, score, weight in top_words:
            direction = "REAL" if score > 0 else "FAKE"
            keywords.append({"word": word, "direction": direction})

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        news_input=news_input,
        keywords=keywords,
        is_uncertain=is_uncertain,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)