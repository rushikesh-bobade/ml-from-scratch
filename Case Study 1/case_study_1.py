# ===============================================================
# Case Study 1: Email Spam Detection Using Machine Learning
# ===============================================================

# -----------------------------
# 1. Import Required Libraries
# -----------------------------
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42

# -----------------------------
# 2. Load Dataset
# -----------------------------
# Download link: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
# Dataset name: spam.csv (you can rename accordingly)

df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
print("Dataset loaded successfully.")
print(df.head())

# -----------------------------
# 3. Data Preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+|www\\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\\s]", "", text)         # remove punctuation/numbers
    text = text.strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
print(f"Total messages: {len(df)}")
print(df['label'].value_counts())

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    df['label'],
    test_size=0.2,
    stratify=df['label'],
    random_state=RANDOM_STATE
)

# -----------------------------
# 5. Model Pipelines
# -----------------------------
models = {
    "Naive Bayes": Pipeline([
        ('tfidf', TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2))),
        ('clf', MultinomialNB())
    ]),
    "Logistic Regression": Pipeline([
        ('tfidf', TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ]),
    "Linear SVM": Pipeline([
        ('tfidf', TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2))),
        ('clf', LinearSVC(random_state=RANDOM_STATE))
    ])
}

# -----------------------------
# 6. Model Training and Evaluation
# -----------------------------
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = classification_report(y_test, preds, output_dict=True)['1']['f1-score']
    results.append((name, acc, f1))

    print(f"--- {name} Results ---")
    print("Accuracy:", round(acc * 100, 2), "%")
    print("F1-score (spam):", round(f1 * 100, 2), "%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))

# -----------------------------
# 7. Compare Model Performance
# -----------------------------
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1-Score'])
print("\nModel Performance Summary:")
print(results_df)

plt.figure(figsize=(7,4))
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.title('Model Comparison (Accuracy)')
plt.ylim(0.9, 1.0)
plt.show()

# -----------------------------
# 8. Select and Save Best Model
# -----------------------------
best_model_name = results_df.sort_values('Accuracy', ascending=False).iloc[0]['Model']
best_pipeline = models[best_model_name]
joblib.dump(best_pipeline, 'best_spam_detector.joblib')
print(f"\n✅ Best model ({best_model_name}) saved as 'best_spam_detector.joblib'")

# -----------------------------
# 9. Predict on New Messages
# -----------------------------
sample_msgs = [
    "Congratulations! You've won a $500 Amazon gift card. Claim now.",
    "Hey, can we meet at 6 pm today?",
    "URGENT! Your account will be suspended unless you verify details now!"
]

preds = best_pipeline.predict(sample_msgs)
for msg, label in zip(sample_msgs, preds):
    print(f"\nMessage: {msg}")
    print("Prediction:", "SPAM" if label == 1 else "HAM")

# -----------------------------
# 10. Optional: ROC-AUC (for models with probability)
# -----------------------------
if hasattr(best_pipeline.named_steps['clf'], "predict_proba"):
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print("\nROC-AUC Score:", round(auc, 4))
