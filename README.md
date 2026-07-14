# 📊 Customer Churn Prediction System

An end-to-end Machine Learning project that predicts whether a customer is likely to churn or stay with a company. The project includes data preprocessing, model training, MLflow experiment tracking, Flask API development, Docker support, and deployment on Render.

---

## 🚀 Live Demo

🌐 **Deployed Application:** https://YOUR_RENDER_URL

---

## 📌 Project Overview

Customer churn prediction helps businesses identify customers who are likely to stop using their services. By predicting churn in advance, companies can improve customer retention and reduce revenue loss.

This project uses Machine Learning techniques to classify customers into:

- ✅ Not Churn
- ❌ Churn

---

## 🎯 Problem Statement

Customer retention is one of the biggest challenges for subscription-based businesses. Acquiring a new customer is much more expensive than retaining an existing one.

This project predicts customer churn using customer demographic and service information, allowing businesses to take preventive actions.

---

# 🛠️ Technologies Used

- Python
- Flask
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Seaborn
- MLflow
- Docker
- Render
- Git & GitHub

---

# 📂 Project Structure

```
Customer_Churn_Project/
│
├── app.py                 # Flask application
├── model.ipynb            # Model training notebook
├── churn_dataset.csv      # Dataset
├── requirements.txt
├── Dockerfile
├── mlruns/                # MLflow experiment tracking
├── templates/
├── static/
└── README.md
```

---

# 📊 Machine Learning Workflow

### 1. Data Collection

- Customer churn dataset

### 2. Data Preprocessing

- Handling missing values
- Label Encoding
- Feature Scaling
- Data Cleaning

### 3. Exploratory Data Analysis

- Correlation Analysis
- Distribution Plots
- Customer Behavior Analysis

### 4. Feature Engineering

- Selecting important features
- Encoding categorical variables

### 5. Model Building

The following algorithms were tested:

- Logistic Regression
- Random Forest
- Decision Tree
- XGBoost

The final deployed model is:

## ✅ XGBoost Classifier

---

# 📈 Model Evaluation

Evaluation metrics used:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# 📊 MLflow Tracking

This project uses **MLflow** to:

- Track experiments
- Compare models
- Save trained models
- Log evaluation metrics

---

# 🌐 Flask Web Application

The application allows users to:

- Enter customer information
- Predict customer churn
- Display prediction result instantly

---

# 🐳 Docker Support

Build Docker Image

```bash
docker build -t customer-churn .
```

Run Docker Container

```bash
docker run -p 5000:5000 customer-churn
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Maruthireddy642/Customer_Churn_Project.git
```

Move into the project directory

```bash
cd Customer_Churn_Project
```

Create Virtual Environment

Linux/Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run the Flask App

```bash
python app.py
```

---

# 📸 Application Preview

(Add your application screenshot here)

Example

```
images/homepage.png
```

---

# 💡 Future Improvements

- Improve prediction accuracy
- Hyperparameter optimization
- Deploy with CI/CD
- User authentication
- Dashboard using Streamlit
- SHAP Explainable AI
- Cloud database integration

---

# 📚 Dataset

Customer Churn Dataset containing customer demographic, account, and service information.

---

# 👨‍💻 Author

**B. Maruthi Reddy**

Final Year B.E. Computer Science & Engineering (Data Science)

GitHub

https://github.com/Maruthireddy642

LinkedIn

(Add your LinkedIn Profile)

---

# ⭐ If you like this project

Please consider giving this repository a ⭐ on GitHub.

---

# 📄 License

This project is developed for educational and learning purposes.
