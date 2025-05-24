import joblib
import os

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl'))
tfidf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf.pkl'))

model = joblib.load(model_path)
tfidf = joblib.load(tfidf_path)

def predict_news(text, threshold=0.6):
    vec = tfidf.transform([text])
    proba = model.predict_proba(vec)[0][1]  # Probability of being FAKE
    label = "Fake" if proba >= threshold else "Real"
    confidence = round(proba * 100, 2) if label == "Fake" else round((1 - proba) * 100, 2)
    return f"{label} ({confidence}% confidence)"
