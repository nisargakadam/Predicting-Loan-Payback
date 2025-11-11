# Loan Default Probability Prediction
[**Quantifying Credit Risk with Logistic Regression**](https://www.kaggle.com/competitions/playground-series-s5e11/overview)


This project predicts the probability that a borrower will repay a loan using customer-level financial and demographic data.  
A logistic regression model was trained to estimate repayment likelihood, achieving an **AUC score of 0.90**, indicating strong distinction between repaid and defaulted loans.

---

## Project Overview
- Built a supervised ML pipeline to model loan repayment probability (binary target).  
- Preprocessed data with one-hot encoding for categorical features and standard scaling for numeric variables.  
- Trained a **Logistic Regression** model (`scikit-learn`) using an ROC-AUC metric for evaluation.  
- Generated **continuous probability predictions** for test data in the required submission format:


---
<pre>
id,loan_paid_back
593994,0.9
593995,0.2
593996,0.1 </pre>

  
---

## Data Features
| Type | Columns |
|------|----------|
| Numeric | annual_income, debt_to_income_ratio, credit_score, loan_amount, interest_rate |
| Categorical | gender, marital_status, education_level, employment_status, loan_purpose, grade_subgrade |
| Target | loan_paid_back (0 = No, 1 = Yes) |

---

## Tools & Libraries
- Python  
- pandas  
- NumPy  
- scikit-learn  
- Jupyter Notebook

---

## Results
| Metric | Score |
|---------|--------|
| Validation ROC-AUC | **0.90** |

The model effectively ranks high- and low-risk borrowers, providing a strong baseline for future ensemble or gradient boosting models.
