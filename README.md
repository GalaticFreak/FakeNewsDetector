![Screenshot 2025-05-24 091452](https://github.com/user-attachments/assets/42866184-3120-4c96-ad1f-683b06e91241)# 📰 Fake News Detector

A machine learning web application that detects whether a news article is **Real** or **Fake**, powered by logistic regression and TF-IDF vectorization.

![screenshot](https://github.com/GalaticFreak/FakeNewsDetector/raw/main/assets/screenshot.png) <!-- Optional image path if you upload a screenshot -->
![Uploading Screenshot 2025-05-24 091452.png![Screenshot 2025-05-21 000853](https://github.com/user-attachments/assets/11a8eb2e-a136-4801-be05-8fcc298cfe89)
![Screenshot 2025-05-24 091526](https://github.com/user-attachments/assets/6c83e115-a8c0-4575-8f33-4fabd96eff49)
…]()

---

## 🚀 Features

- Detects fake vs real news articles with high accuracy (98%+)
- Shows prediction **confidence percentage**
- Clean, animated UI built with **HTML/CSS**
- Python backend using **Flask**
- Custom-trained ML model using **Logistic Regression** + **TF-IDF**

---

## 🛠️ Tech Stack

| Frontend       | Backend    | ML Model        | Tools      |
|----------------|------------|-----------------|------------|
| HTML5, CSS3    | Flask      | LogisticRegression | Scikit-learn |
| Jinja2 Templating | Python 3 | TF-IDF Vectorizer | Joblib      |

---

## ⚙️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/GalaticFreak/FakeNewsDetector.git
cd FakeNewsDetector
pip install -r requirements.txt
python src/train_model.py
cd webapp
python app.py

