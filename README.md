# ml-project-credit-risk-modeling
Credit Risk Prediction Project

Credit Risk Modelling
🔗 Live App:

👉 https://ml-project-credit-risk-modeling-piyush.streamlit.app/

A complete end-to-end Credit Risk Prediction System built using Machine Learning and Streamlit.
The system predicts default probability, generates a credit score (300–900), and provides a credit rating using a trained ML model.

This project simulates how real-world financial institutions evaluate borrower risk.

# 🚀 Key Features

🔍 Predicts default probability using a trained ML model

📈 Generates a credit score (300–900)

🏷️ Classifies borrowers into Poor / Average / Good / Excellent

📊 Displays Loan-to-Income ratio

✨ Clean Streamlit UI for interactive usage

📦 Deployment-ready model artifact (model_data.joblib)

⚙️ Fully reproducible pipeline using preprocessing + scaling + one-hot encoding

🧠 Model Training Workflow

The credit risk model was trained using a structured and industry-aligned ML workflow.

# 1️⃣ Data Understanding & EDA

✅Explored borrower demographics, loan attributes, credit behavior

✅Analyzed delinquency indicators, loan purpose, income groups, risk patterns

✅Checked missing values, outliers, skewness, and class imbalance

✅Performed correlation and feature importance insights

# 2️⃣ Data Preprocessing

👉Missing value treatment using median/mode

👉Outlier handling using IQR and percentile capping

👉One-hot encoding for categorical variables

👉Scaling of continuous variables using StandardScaler

# Feature engineering:

👉Loan-to-Income Ratio

👉Credit Utilization Ratio

👉Delinquency Ratio

👉DPD (Days Past Due) Metrics

Risk-based ratios derived from credit behavior

# 3️⃣ Train–Test Split

Used Stratified Split to preserve default distribution

Prevented data leakage by fitting scaler & encoder only on training data

# 4️⃣ Model Building & Comparison

## Multiple algorithms were evaluated:

Logistic Regression

Random Forest

XGBoost

Gradient Boosting

Metrics used:

AUC-ROC (primary metric)

Precision–Recall Curve

F1 Score

Brier Score (probability calibration)

# 5️⃣ Hyperparameter Optimization (Optuna)

Implemented Optuna for automated hyperparameter optimization

Objective function optimized ROC-AUC

Tuned:

Number of estimators

Max depth

Learning rate

Regularization parameters

Achieved faster and more accurate tuning compared to manual grid/random search

# 6️⃣ Final Model Selection

Chose the best performing model based on AUC and calibration

Exported the following into model_data.joblib:

model

scaler

features

cols_to_scale

# 7️⃣ Credit Score Mapping

Default probability → Credit Score (300–900)

Default Probability	Credit Score	Rating
> 0.7	300–499	Poor
0.4–0.7	500–649	Average
0.2–0.4	650–749	Good
< 0.2	750–900	Excellent

This mapping mimics real credit bureau behavior.

# 8️⃣ Deployment Preparation

Serialized artifacts using joblib

Ensured feature ordering matches the trained model

Integrated with prediction_helper.py

Deployed frontend using Streamlit Cloud

# 🖥️ UI Workflow
## 1️⃣ User enters details:

Age

Income

Loan amount

Tenure

Credit utilization

Delinquency metrics

Residence type

Loan purpose

Loan type

## 2️⃣ Model processes:

Scales numeric features

Encodes categorical variables

Computes engineered features

Predicts using trained model

## 3️⃣ Output includes:

Default probability

Credit score

Credit rating

Loan-to-income ratio

# ⚙️ Tech Stack
Backend / ML

Python

NumPy

Pandas

Scikit-learn

Optuna (for hyperparameter tuning)

Joblib (for model artifact storage)

Frontend

Streamlit

Deployment

Streamlit Cloud

# 🛠️ Running the Project Locally
## Step 1: Clone the repo
git clone https://github.com/piyushpanwar/ml-project-credit-risk-modeling.git
cd ml-project-credit-risk-modeling

## Step 2: Create virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
#### OR
.\.venv\Scripts\activate        # Windows

## Step 3: Install dependencies
pip install -r requirements.txt

## Step 4: Run the app
streamlit run main.py

# 👨‍💻 Author: Piyush Panwar

Passionate Data Science & ML Engineer
💼 Building predictive systems across Finance, Healthcare, and Real-time ML
🔗 GitHub: https://github.com/piyushpanwar

🔗 LinkedIn: https://www.linkedin.com/in/piyushpanwar/
