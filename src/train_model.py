import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
df_fake = pd.read_csv("dataset/Fake.csv")
df_fake["label"] = 1  # 1 = Fake

df_real = pd.read_csv("dataset/True.csv")
df_real["label"] = 0  # 0 = Real

# Combine and balance
df = pd.concat([df_fake, df_real])
df_fake = df[df["label"] == 1]
df_real = df[df["label"] == 0]
min_len = min(len(df_fake), len(df_real))
df = pd.concat([df_fake.sample(min_len), df_real.sample(min_len)])
df = df.sample(frac=1).reset_index(drop=True)

# Combine title and text
df["combined"] = df["title"].fillna('') + " " + df["text"].fillna('')
X = df["combined"]
y = df["label"]

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5, max_features=10000, ngram_range=(1, 2))
X_vec = tfidf.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model and vectorizer
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/model.pkl")
joblib.dump(tfidf, "../models/tfidf.pkl")

# Sample prediction debug
sample_real = "The WHO declared the end of the COVID-19 emergency."
sample_fake = "Aliens opened a pizza shop in Antarctica."
vec_real = tfidf.transform([sample_real])
vec_fake = tfidf.transform([sample_fake])

print("Prediction (REAL):", model.predict(vec_real))
print("Prediction (FAKE):", model.predict(vec_fake))
print("Probabilities (REAL):", model.predict_proba(vec_real))
print("Probabilities (FAKE):", model.predict_proba(vec_fake))
