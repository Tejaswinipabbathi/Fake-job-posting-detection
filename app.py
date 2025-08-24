import re
import pickle
from flask import Flask, render_template, request

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    explanation = []

    if request.method == "POST":
        text = request.form["job_text"]
        clean = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())[:3000]
        vec = vectorizer.transform([clean])
        prob = model.predict_proba(vec)[0][1]
        prediction = "🚨 FAKE" if prob >= 0.5 else "✅ REAL"
        confidence = round(prob if prediction == "🚨 FAKE" else 1 - prob, 2)

        # Simple explanation rules
        lowered = text.lower()
        if "from home" in lowered or "no experience" in lowered:
            explanation.append("Too good to be true offers like 'work from home' or 'no experience needed'.")
        if "$" in text or "earn" in lowered:
            explanation.append("Mentions of high earnings or money upfront.")
        if "urgent" in lowered or "limited spots" in lowered:
            explanation.append("Urgency or scarcity tactics may indicate fraud.")
        if len(explanation) == 0:
            explanation.append("The language and structure match patterns seen in fake posts.")

    return render_template("index.html", prediction=prediction, confidence=confidence, explanation=explanation)

if __name__ == "__main__":
    app.run(debug=True)
