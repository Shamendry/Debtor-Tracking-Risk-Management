# Debtor-Tracking-Risk-Management
**Project Overview**

This project implements a data driven debtor risk management framework that combines customer behaviour analysis, clustering, and predictive machine learning models using historical sales and customer outstanding data.

Rather than focusing only on credit risk prediction, the project analyses how customers behave, how valuable and loyal they are, and how those behavioural patterns relate to payment risk. This enables a shift from reactive credit control to proactive, behaviour informed risk management.

All datasets used are commercially confidential and are therefore excluded from this public repository. The notebooks demonstrate the full methodology and logic.

**Business Problem**

Traditional credit management relies heavily on overdue reports and static aging analysis, which only highlight issues after they occur. This project aims to:

Identify risky customers early

Understand customer purchasing and payment behaviour

Segment customers into meaningful behavioural groups

Predict credit risk bands using historical behaviour and outstanding patterns

Data Sources (Anonymised)

The project uses two primary datasets:

Sales Transactions Data
Invoice level sales records containing product, value, quantity, and date information.

Customer Outstanding Data
Open invoices, overdue days, receipts, debit and credit notes, and outstanding balances.

Customer and product identifiers are anonymised.

**Project Structure**
<img width="1536" height="1024" alt="ChatGPT Image Feb 4, 2026, 08_49_05 AM" src="https://github.com/user-attachments/assets/db23859f-96a8-4db0-9fc6-aa4f6495ca90" />


**Methodology**
1. Data Preparation

Cleaning and normalisation of column names

Handling ERP date artifacts and invalid timestamps

Validation of invoice and due dates

Removal of inconsistent and incomplete records

Separation of transactional and customer level data


2. Customer Behaviour Feature Engineering

Customer behaviour is derived from transactional sales data and aggregated at customer level.

Key behavioural features include:

Total sales value

Average invoice value

Invoice count

Active purchasing months

Relationship length in years

Recency of last purchase

Purchase frequency

Number of unique products purchased

These features capture customer size, loyalty, consistency, and engagement.


3. Behavioural Clustering

Feature scaling using StandardScaler

K Means clustering applied to behavioural features

Optimal number of clusters evaluated using:

Elbow Method

Silhouette Score

Clusters interpreted and labelled using business logic

Example behavioural segments:

New Customers

Dormant or At Risk Customers

Loyal Medium Buyers

High Frequency Low Value Customers

Very High Value Strategic Customers


4. Debtor Risk Feature Engineering

From customer outstanding data:

Invoice, receipt, debit note, and credit note counts

Average and maximum overdue days

Outstanding balances in LKR and USD

Receipt to invoice ratios

These features capture payment discipline and exposure risk.


5. Risk Modelling

Two modelling approaches are implemented:

Binary Classification

High Risk vs Non High Risk customers

Multi Class Classification

High Risk

Medium Risk

Low Risk

Models used:

Logistic Regression

Random Forest Classifier


6. Temporal Validation

To reflect real world deployment:

Models are trained on historical data

Tested on future unseen periods

Prevents data leakage

Produces realistic performance estimates


7. Model Evaluation and Explainability

Confusion matrices

Precision, recall, and F1 score

Feature importance analysis

Key drivers of risk include:

Average overdue days

Maximum overdue days

Invoice count

Outstanding balances

Technologies Used

Python

Pandas and NumPy

Scikit learn

Matplotlib

Jupyter Notebook

**Data Confidentiality**

All raw data files are excluded due to confidentiality.

The notebooks are structured so that:

Data paths can be easily replaced

Expected column names are visible

Logic is fully reproducible with similar datasets

**How to Use**

Clone the repository

Review notebooks in sequence

Replace data loading paths with your own datasets

Execute feature engineering, clustering, and modelling steps

Academic and Practical Value

**This project demonstrates:**

End to end analytics workflow

Practical application of clustering and machine learning

Business oriented interpretation of results

Realistic modelling using temporal validation

Author

Shamendry
Business Data Analytics Undergraduate

Focus areas: Credit Risk Analytics, Machine Learning, Business Intelligence
