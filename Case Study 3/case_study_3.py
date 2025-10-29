# ===============================================================
# Case Study 3: Handwritten Digit Recognition using SVM and KNN
# ===============================================================

# -----------------------------
# 1. Import Libraries
# -----------------------------
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

RANDOM_STATE = 42

# -----------------------------
# 2. Load Dataset
# -----------------------------
digits = load_digits()
X = digits.data
y = digits.target

print("Dataset loaded successfully.")
print("Shape:", X.shape)
print("Classes:", np.unique(y))

# Visualize some digits
fig, axes = plt.subplots(2, 5, figsize=(8, 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Label: {digits.target[i]}")
    ax.axis('off')
plt.suptitle("Sample Handwritten Digits")
plt.show()

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# -----------------------------
# 4. Define Pipelines
# -----------------------------
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=30, random_state=RANDOM_STATE)),
    ('svc', SVC())
])

pipe_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=30, random_state=RANDOM_STATE)),
    ('knn', KNeighborsClassifier())
])

# -----------------------------
# 5. Hyperparameter Tuning
# -----------------------------
param_grid_svm = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 0.01, 0.001],
    'svc__kernel': ['rbf']
}

param_grid_knn = {
    'knn__n_neighbors': [3, 5, 7],
    'knn__weights': ['uniform', 'distance']
}

# -----------------------------
# 6. Grid Search for SVM
# -----------------------------
print("\nTuning SVM...")
start = time.time()
grid_svm = GridSearchCV(pipe_svm, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_svm.fit(X_train, y_train)
svm_time = time.time() - start
print(f"SVM Best Params: {grid_svm.best_params_}")
print(f"SVM Best CV Accuracy: {grid_svm.best_score_:.4f}")

# -----------------------------
# 7. Grid Search for KNN
# -----------------------------
print("\nTuning KNN...")
start = time.time()
grid_knn = GridSearchCV(pipe_knn, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
grid_knn.fit(X_train, y_train)
knn_time = time.time() - start
print(f"KNN Best Params: {grid_knn.best_params_}")
print(f"KNN Best CV Accuracy: {grid_knn.best_score_:.4f}")

# -----------------------------
# 8. Evaluation on Test Set
# -----------------------------
models = {
    "SVM": grid_svm.best_estimator_,
    "KNN": grid_knn.best_estimator_
}

results = []
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc))
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# -----------------------------
# 9. Model Comparison
# -----------------------------
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy'])
print("\nModel Comparison:")
print(results_df)

plt.figure(figsize=(6, 4))
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.title('Model Accuracy Comparison')
plt.ylim(0.9, 1.0)
plt.show()

# -----------------------------
# 10. Save Best Model
# -----------------------------
best_model_name = results_df.sort_values('Accuracy', ascending=False).iloc[0]['Model']
best_model = models[best_model_name]
joblib.dump(best_model, 'best_digit_recognizer.joblib')
print(f"\n✅ Best model ({best_model_name}) saved as 'best_digit_recognizer.joblib'")

# -----------------------------
# 11. Inference on Sample Digits
# -----------------------------
sample_idx = np.random.randint(0, len(X_test), 5)
sample_images = X_test[sample_idx]
sample_labels = y_test[sample_idx]
predictions = best_model.predict(sample_images)

plt.figure(figsize=(10, 3))
for i, idx in enumerate(sample_idx):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
    plt.title(f"Pred: {predictions[i]}\nTrue: {sample_labels[i]}")
    plt.axis('off')
plt.suptitle("Predictions on Sample Digits")
plt.show()
