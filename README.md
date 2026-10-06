# 📉 Debtor Risk Management & Credit Intelligence System

<p align="center">
  <strong>An end-to-end Machine Learning and Business Intelligence solution for identifying high-risk debtors, analysing financial exposure, and supporting proactive credit-management decisions.</strong>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Analytics-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-185FA5?style=for-the-badge)

</p>

---

## 🚀 Live Dashboard

### [▶ Launch Debtor Risk Management Dashboard](https://debtorriskmanagement.streamlit.app/)

The interactive dashboard provides portfolio-level risk monitoring together with detailed customer-level credit intelligence.

---

## 📌 Project Overview

Managing debtor exposure becomes increasingly difficult as customer portfolios grow. Traditional credit analysis often relies on static ageing reports and manually defined thresholds, making it difficult to identify emerging risks early.

This project develops a **data-driven Debtor Risk Management System** that combines Machine Learning, behavioural segmentation and interactive Business Intelligence to identify potentially risky accounts and explain the factors contributing to their risk.

The solution transforms historical customer, invoice, payment and purchasing behaviour into actionable credit intelligence.

Instead of simply answering:

> **"How much does this customer owe?"**

the system also helps answer:

> **"How risky is this customer, why are they risky, and where should the credit team focus first?"**

---

# 🎯 Business Objectives

The system was designed to support credit and finance teams by enabling them to:

- Identify **high-risk debtor accounts**
- Prioritise accounts requiring immediate attention
- Measure financial exposure by risk level
- Detect problematic payment and purchasing behaviour
- Segment customers based on behavioural characteristics
- Understand the factors driving individual risk predictions
- Compare debtor exposure across countries and customer groups
- Analyse overdue balances and product exposure
- Support data-driven credit-control decisions

---

# 📊 Portfolio Snapshot

Based on the processed dataset included in the project:

| Metric | Result |
|---|---:|
| 👥 Total Customers | **656** |
| 🔴 High Risk Customers | **332** |
| 🟠 Medium Risk Customers | **155** |
| 🟢 Low Risk Customers | **169** |
| 💰 Total Outstanding | **$28.69M** |
| ⚠️ High Risk Outstanding | **$14.07M** |
| 📅 Average Overdue Period | **182 Days** |
| 🧠 Model Features | **20** |
| 🎯 ROC-AUC | **86.45%** |
| ✅ Cross-Validation F1 | **87.04%** |

---

# 🧠 Machine Learning Architecture

The project combines **supervised learning, unsupervised learning and explainable AI**.

```text
                    ┌────────────────────────────┐
                    │      Raw Debtor Data       │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ Data Cleaning & Processing │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │    Feature Engineering     │
                    │      20+ Risk Signals      │
                    └─────────────┬──────────────┘
                                  │
                   ┌──────────────┴───────────────┐
                   ▼                              ▼
        ┌─────────────────────┐       ┌─────────────────────┐
        │ Random Forest Model │       │   K-Means Model     │
        │   Risk Prediction   │       │ Customer Segments   │
        └──────────┬──────────┘       └──────────┬──────────┘
                   │                              │
                   └──────────────┬───────────────┘
                                  ▼
                    ┌────────────────────────────┐
                    │     SHAP Explainability    │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ Streamlit Risk Dashboard   │
                    └────────────────────────────┘
```

---

# 🔍 Risk Indicators

The predictive model evaluates customers using a range of financial and behavioural characteristics, including:

### Financial Exposure

- Total invoices outstanding
- Average invoice value
- Total outstanding balance
- Debit note amount
- Credit note activity
- Lifetime sales value

### Payment Behaviour

- Payments received
- Payment-to-invoice ratio
- Unallocated payments
- Unapplied receipt behaviour

### Customer Behaviour

- Days since last purchase
- Purchase frequency
- Active purchasing months
- Customer relationship duration
- Number of products purchased
- Product category diversity

### Behavioural Intelligence

- Customer behavioural segment
- Category concentration
- Purchase activity patterns

Together, these features provide a broader view of debtor risk than outstanding balances alone.

---

# 🤖 Risk Prediction Model

A **Random Forest classifier** is used to categorise customers into:

🟢 **Low Risk**

🟠 **Medium Risk**

🔴 **High Risk**

The model achieved:

```text
ROC-AUC                    0.8645
Cross-Validation F1       0.8704
Predictive Features       20
```

The resulting probability outputs are transformed into an intuitive **0–100 risk score**, making model outputs easier for business users to interpret.

---

# 🔬 Explainable AI with SHAP

A credit-risk prediction becomes significantly more useful when the user can understand **why** the model produced the prediction.

The project therefore integrates **SHAP (SHapley Additive exPlanations)**.

SHAP analysis is used to:

- Identify the strongest drivers of customer risk
- Explain individual customer predictions
- Understand global model behaviour
- Distinguish factors increasing and decreasing risk

This makes the model significantly more transparent to non-technical users and improves its usefulness for real-world credit decisions.

### SHAP Feature Importance

![SHAP Summary](Debtor%20Risk%20Management/data_processed/shap_summary_high.png)

---

# 👥 Customer Behaviour Segmentation

Risk alone does not fully describe customer behaviour.

The project therefore uses **K-Means clustering** to group customers according to behavioural characteristics.

This enables the credit team to examine risk within different customer profiles rather than treating every debtor in the same way.

### Customer Segmentation

![Customer Segments](Debtor%20Risk%20Management/data_processed/cluster_pca_plot.png)

The segmentation approach enables analysis of:

- Customer activity patterns
- Purchasing frequency
- Customer lifecycle behaviour
- Product diversity
- Financial exposure
- Risk concentration by segment

---

# 📊 Interactive Streamlit Dashboard

The analytical outputs are delivered through an interactive Streamlit application.

The dashboard contains three primary modules.

## 1️⃣ Risk Intelligence Overview

Provides an executive-level overview of the entire debtor portfolio.

Key functionality includes:

- Total customer exposure
- High / Medium / Low risk distribution
- Outstanding balance by risk category
- High-risk financial exposure
- Top high-risk accounts
- Global geographic exposure
- Country-level average risk
- Downloadable high-risk debtor list

---

## 2️⃣ Customer Account Review

Allows individual debtor accounts to be investigated in detail.

Users can search for a customer and view:

- Customer risk classification
- 0–100 risk score
- Customer profile
- Outstanding balance
- Model prediction probabilities
- SHAP risk drivers
- Invoice behaviour
- Payment behaviour
- Purchase history
- Customer behavioural segment
- Peer comparisons

This transforms the model from a portfolio-level prediction tool into an operational **credit-review system**.

---

## 3️⃣ Segments & Financial Exposure

Provides deeper portfolio analytics across customer groups.

Analysis includes:

- Customer behavioural clusters
- Cluster risk distribution
- Segment financial exposure
- Product/category exposure
- Overdue balance analysis
- Payment behaviour
- Risk concentration
- Customer segment comparisons

---

# 🌍 Geographic Risk Intelligence

The dashboard contains interactive geographic visualisations showing debtor exposure across different countries.

Countries can be analysed using:

- Total outstanding balance
- Number of customers
- Average customer risk
- Number of high-risk debtors

This provides a management-level view of where portfolio exposure is geographically concentrated.

---

# 📈 Model Evaluation

The repository includes several evaluation and diagnostic outputs.

### Confusion Matrix

![Confusion Matrix](Debtor%20Risk%20Management/data_processed/confusion_matrix_rf.png)

### Feature Importance

![Feature Importance](Debtor%20Risk%20Management/data_processed/feature_importance.png)

### K-Means Elbow Analysis

![Elbow Plot](Debtor%20Risk%20Management/data_processed/elbow_plot.png)

### Silhouette Analysis

![Silhouette Plot](Debtor%20Risk%20Management/data_processed/silhouette_plot.png)

---

# 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Classification | Random Forest |
| Clustering | K-Means |
| Explainable AI | SHAP |
| Visualisation | Plotly |
| Dashboard | Streamlit |
| Model Persistence | Joblib |
| Development | Jupyter Notebook |
| Version Control | Git & GitHub |

---

# 📁 Project Structure

```text
Debtor-Tracking-Risk-Management/
│
├── Debtor Risk Management/
│   │
│   ├── Dashboard/
│   │   └── app_1.py
│   │
│   ├── data_raw/
│   │   └── Debtor Project.ipynb
│   │
│   ├── data_processed/
│   │   ├── behaviour_clusters.csv
│   │   ├── final_risk_output.csv
│   │   ├── master_features.csv
│   │   ├── shap_values_all_customers.csv
│   │   ├── kpi_summary.json
│   │   └── model visualisations
│   │
│   └── model_artifacts/
│       ├── rf_risk_band_model.pkl
│       ├── kmeans_model.pkl
│       ├── behaviour_scaler.pkl
│       ├── feature_columns.pkl
│       ├── risk_band_label_encoder.pkl
│       └── shap_explainer.pkl
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/Debtor-Tracking-Risk-Management.git
```

### 2. Navigate to the repository

```bash
cd Debtor-Tracking-Risk-Management
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 5. Install the dependencies

```bash
pip install -r requirements.txt
```

### 6. Start the Streamlit application

```bash
streamlit run "Debtor Risk Management/Dashboard/app_1.py"
```

The dashboard should then open automatically in your browser.

---

# 📦 Python Dependencies

The project currently uses:

```text
streamlit
pandas
numpy
plotly
scikit-learn
xgboost
shap
joblib
openpyxl
```

---

# 💡 Key Project Strengths

### End-to-End Analytics

The project covers the complete workflow from data preparation and feature engineering to Machine Learning, explainability and dashboard deployment.

### Business-Oriented Machine Learning

Predictions are presented as interpretable risk scores and portfolio insights rather than raw algorithm outputs.

### Explainable Predictions

SHAP enables users to understand the reasons behind each prediction.

### Behaviour + Financial Risk

Customer risk is evaluated using both financial exposure and purchasing/payment behaviour.

### Interactive Decision Support

The Streamlit interface enables finance users to investigate individual accounts without requiring programming knowledge.

---

# 🚧 Potential Future Development

The system could be extended with:

- Live ERP / accounting-system integration
- Automated daily data refreshes
- Automated email alerts for newly identified high-risk customers
- Customer risk movement tracking
- Historical risk-score trends
- Expected-loss modelling
- Probability-of-default modelling
- Credit-limit recommendations
- Automated collection prioritisation
- Role-based dashboard access
- Cloud database integration
- API-based model serving

---

# ⚠️ Disclaimer

This project is intended for analytical, educational and decision-support purposes.

Machine Learning predictions should complement rather than replace professional credit assessment and organisational credit-control procedures.

---

# 👨‍💻 Author

Shamendry Stephen

Business Data Analytics | Machine Learning | Business Intelligence | Data Analytics

[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?style=for-the-badge&logo=github)](https://github.com/Shamendry/Debtor-Tracking-Risk-Management)

---

<p align="center">
  <strong>Built with Python, Machine Learning, Explainable AI and Streamlit.</strong>
</p>
