import os

import matplotlib

matplotlib.use('QtAgg')

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# =========================================================
# CREATE RESULTS FOLDER
# =========================================================

results_folder = "results"

os.makedirs(results_folder, exist_ok=True)

print(f"Results folder created: {results_folder}")


# =========================================================
# LOAD DATASET
# =========================================================

digits = load_digits()

print("\nDataset Keys:")
print(digits.keys())

print("\nData Shape:", digits.data.shape)
print("Images Shape:", digits.images.shape)
print("Target Shape:", digits.target.shape)


# =========================================================
# CREATE FEATURES AND LABELS
# =========================================================

X = digits.data
y = digits.target


# =========================================================
# DISPLAY FIRST 50 IMAGES
# =========================================================

fig1, axes = plt.subplots(5, 10, figsize=(12, 6))

for i, ax in enumerate(axes.flat):

    ax.imshow(digits.data[i].reshape(8, 8), cmap='gray')

    ax.set_title(f"{y[i]}")

    ax.axis('off')

plt.tight_layout()

digits_path = os.path.join(results_folder, "first_50_digits.png")

fig1.savefig(digits_path)

print(f"\nFirst 50 digits image saved to:\n{digits_path}")


# =========================================================
# TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


# =========================================================
# STANDARDIZATION
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# =========================================================
# CREATE AND TRAIN KNN MODEL
# =========================================================

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train_scaled, y_train)

print("\nKNN Model Trained Successfully")


# =========================================================
# PREDICTIONS
# =========================================================

y_pred = knn.predict(X_test_scaled)


# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:\n")

report = classification_report(y_test, y_pred)

print(report)

report_path = os.path.join(results_folder, "classification_report.txt")

with open(report_path, "w") as f:
    f.write(report)

print(f"\nClassification report saved to:\n{report_path}")


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_test, y_pred)

fig2, ax = plt.subplots(figsize=(8, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=digits.target_names
)

disp.plot(ax=ax)

plt.title("Confusion Matrix")

cm_path = os.path.join(results_folder, "confusion_matrix.png")

fig2.savefig(cm_path)

print(f"\nConfusion matrix saved to:\n{cm_path}")


# =========================================================
# SHOW SAMPLE PREDICTIONS
# =========================================================

fig3, axes = plt.subplots(2, 5, figsize=(10, 5))

sample_indices = np.arange(10)

for ax, idx in zip(axes.flat, sample_indices):

    ax.imshow(X_test[idx].reshape(8, 8), cmap='gray')

    ax.set_title(
        f"Pred: {y_pred[idx]}\nTrue: {y_test[idx]}"
    )

    ax.axis('off')

plt.tight_layout()

predictions_path = os.path.join(
    results_folder,
    "sample_predictions.png"
)

fig3.savefig(predictions_path)

print(f"\nSample predictions saved to:\n{predictions_path}")


# =========================================================
# ACCURACY BAR GRAPH
# =========================================================

fig4 = plt.figure(figsize=(5, 5))

plt.bar(["KNN Accuracy"], [accuracy])

plt.ylim(0, 1)

plt.ylabel("Accuracy")

plt.title("Model Accuracy")

accuracy_path = os.path.join(
    results_folder,
    "accuracy_graph.png"
)

fig4.savefig(accuracy_path)

print(f"\nAccuracy graph saved to:\n{accuracy_path}")


# =========================================================
# SHOW ALL FIGURES
# =========================================================

plt.show()