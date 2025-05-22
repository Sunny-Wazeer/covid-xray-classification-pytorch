# 💳 Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning. It includes data preprocessing, class balancing using SMOTE, model training with three algorithms, and evaluation using key metrics.

## 📊 Dataset

The dataset used is the popular [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. It contains **284,807 transactions**, of which only **492 are fraud**.

## ⚙️ Workflow

1. **Data Loading and Cleaning**
2. **Class Distribution Analysis**
3. **Correlation Heatmap**
4. **Data Balancing**
5. **Train-Test Splitting**
6. **Model Pipelines** with:
    - Standard Scaler
    - SMOTE Oversampling
    - ML Classifier
7. **Model Evaluation**
8. **Model Saving with Joblib**

## 🤖 Models Used

- Logistic Regression
- Random Forest
- XGBoost

## 🧪 Evaluation Metrics

Each model is evaluated using:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC AUC Score
- Accuracy

## 📦 Requirements

See [`requirements.txt`](./requirements.txt)

## 💾 Save & Load Model

```python
import joblib
joblib.dump(model, "model.pkl")

# Load
model = joblib.load("model.pkl")
```

## 📂 Project Structure

```
credit-card-fraud-detection/
├── creditcard.csv                  # Dataset (not included in repo)
├── fraud_detection.ipynb           # Full notebook/script
├── random_forest_model.pkl         # Trained model file
├── README.md
└── requirements.txt
```

## 🛠️ Future Work

- Use ensemble voting of top models
- Hyperparameter tuning (GridSearchCV)
- Add a Flask or Streamlit interface for real-time prediction

## 📜 License

MIT License