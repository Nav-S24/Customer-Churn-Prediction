import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 1: Load the dataset
data = pd.read_csv('Telco-Customer-Churn.csv')

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
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

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
