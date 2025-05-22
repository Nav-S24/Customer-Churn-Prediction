# 📉 Customer Churn Prediction

This project predicts whether a customer is likely to churn using machine learning. It uses a public telecom dataset to build a classification model that helps businesses proactively retain their customers.

---

## 📌 Project Objectives

- Identify customers likely to churn using historical usage data.
- Perform end-to-end preprocessing including missing value handling, encoding, and feature scaling.
- Train and evaluate a supervised ML model for classification.
- Analyze feature importance to understand key churn drivers.

---

## 🧠 Machine Learning Approach

- **Algorithm**: Random Forest Classifier  
- **Target**: `Churn` (1 = Yes, 0 = No)  
- **Features Used**: Demographics, contract type, usage behavior, etc.  
- **Tools/Libraries**: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

---

## 🔍 Dataset

- **Source**: [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size**: ~7,000 customer records
- **Attributes**: gender, tenure, monthly charges, internet service, etc.

---

## 🛠️ Project Workflow

1. **Data Cleaning**  
   - Removed missing values  
   - Handled categorical features with one-hot encoding  

2. **Feature Engineering**  
   - Scaled numerical columns using `StandardScaler`  
   - Converted `Churn` column to binary format  

3. **Model Training & Evaluation**  
   - Used `train_test_split()` for validation  
   - Trained a `RandomForestClassifier`  
   - Evaluated using Accuracy, Precision, Recall, F1 Score  

4. **Insights & Visualizations**  
   - Boxplot of `tenure` vs `churn`  
   - Feature importance bar chart  

---

## 📊 Model Performance

- **Accuracy**: 0.8034
- **F1 Score**: 0.5496 
- **Top Features**: tenure, contract type, monthly charges

---

## 📎 File Structure

```bash
customer-churn/
│
├── churn_prediction.ipynb   # Jupyter notebook with code
├── Telco-Customer-Churn.csv # Dataset
├── README.md                # Project description
└── churn_model.pkl          # (Optional) Saved model for deployment
