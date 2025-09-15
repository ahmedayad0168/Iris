# ===============================
# Import Libraries
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===============================
# Load Data
# ===============================
# Option 1: load from sklearn datasets
# from sklearn.datasets import load_iris
# iris = load_iris()
# data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Option 2: load from CSV
data = pd.read_csv('D:/Learn Machine Learning/datasets/Classification/Iris.csv')

# ===============================
# Split Features & Labels
# ===============================
X = data.drop(['label'], axis=1)
y = data['label']

# ===============================
# Train/Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.80, random_state=42
)

# ===============================
# Define Models
# ===============================
model_logistic = LogisticRegression(max_iter=200, random_state=42)   
model_rf = RandomForestClassifier(random_state=42)

# ===============================
# Train Models
# ===============================
model_logistic.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

# ===============================
# Evaluate Models (Train/Test Scores)
# ===============================
print("="*50)
print(" Logistic Regression ".center(50, "-"))
print(f"Train Score: {model_logistic.score(X_train, y_train):.3f}")
print(f"Test  Score: {model_logistic.score(X_test, y_test):.3f}")

print("="*50)
print(" Random Forest ".center(50, "-"))
print(f"Train Score: {model_rf.score(X_train, y_train):.3f}")
print(f"Test  Score: {model_rf.score(X_test, y_test):.3f}")

# ===============================
# Predictions on Full Dataset
# ===============================
y_pred_logistic = model_logistic.predict(X)
y_pred_rf = model_rf.predict(X)

# ===============================
# Metrics (Accuracy, Confusion Matrix, Report)
# ===============================
print("="*50)
print(" Logistic Regression Metrics ".center(50, "-"))
print("Accuracy:", accuracy_score(y, y_pred_logistic))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred_logistic))
print("Classification Report:\n", classification_report(y, y_pred_logistic))

print("="*50)
print(" Random Forest Metrics ".center(50, "-"))
print("Accuracy:", accuracy_score(y, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred_rf))
print("Classification Report:\n", classification_report(y, y_pred_rf))

# ===============================
# Visualization
# ===============================
# Pairplot
sns.pairplot(data, hue="label", diag_kind="kde")
plt.show()

# Boxplots
plt.figure(figsize=(12, 8))
for i, col in enumerate(data.columns[:-1]):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x="label", y=col, data=data)
    plt.title(col)
plt.tight_layout()
plt.show()

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.drop("label", axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# ===============================
# Cross Validation
# ===============================
# Logistic Regression CV
scores_logistic = cross_val_score(model_logistic, X, y, cv=5)  # 5-fold
print("="*50)
print(" Logistic Regression Cross Validation ".center(50, "-"))
print("Scores:", scores_logistic)
print("Mean Accuracy:", scores_logistic.mean())

# Random Forest CV
scores_rf = cross_val_score(model_rf, X, y, cv=5)  # 5-fold
print("="*50)
print(" Random Forest Cross Validation ".center(50, "-"))
print("Scores:", scores_rf)
print("Mean Accuracy:", scores_rf.mean())
