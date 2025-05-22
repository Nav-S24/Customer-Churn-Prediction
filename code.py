import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Step 1: Load the dataset
data = pd.read_csv('TelcoCustomerChurn.csv')

# Keep a copy of original data for EDA
data_orig = data.copy()

# Step 2: Data Preprocessing
# Drop missing values
data.dropna(inplace=True)

# Convert target column to binary (Yes → 1, No → 0)
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical columns (excluding target)
categorical_cols = data.select_dtypes(include=['object']).columns.drop('Churn', errors='ignore')
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Scale numerical columns (excluding target)
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.drop('Churn', errors='ignore')
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Step 3: Visualize Tenure vs Churn (before scaling)
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_orig, x='Churn', y='tenure')
plt.title('Tenure vs Churn')
plt.show()

# Step 4: Train-Test Split
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
#y_pred = model.predict(X_test)
#print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print Metrics
print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1score:.4f}")

# Optional: Display confusion matrix as heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()


# Step 7: Plot Feature Importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort feature importances descending

# Get top 15 features
top_n = 15
top_features = X.columns[indices][:top_n]
top_importances = importances[indices][:top_n]

plt.figure(figsize=(12, 6))
plt.title('Top 15 Feature Importances')
sns.barplot(x=top_importances, y=top_features, palette='viridis')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
