#  Customer Churn Prediction

A machine learning project that predicts which telecom customers are likely to leave, and suggests what a company can do to keep them. It includes a full data analysis notebook and an interactive **Streamlit app**.

**🔗 Live app:** [https://customer-churn-prediction-esmabhde7rvmazjk6pkjrd.streamlit.app/]

---

## Why this project?

Getting a new customer costs much more than keeping an existing one. If a company knows who is about to leave, it can offer a discount or a call in time. This project finds those customers and helps decide how much money to spend on keeping them.

## Dataset

- **IBM Telco Customer Churn dataset**
- Each row is one customer: their services, contract, charges and whether they left (churn).


## What I did

1. **Data collection and cleaning:** fixed data quality problems, such as blank values in `TotalCharges`.
2. **Exploratory data analysis (EDA):** statistical summary and interactive **Plotly** charts to see what drives churn.
3. **Feature engineering:** encoding, scaling and a train-test split.
4. **Handling imbalance:** used **SMOTE** to balance the classes. It was applied only to the training folds, to avoid data leakage.
5. **Modeling:** built and compared **Logistic Regression** and **Random Forest** in an sklearn/imblearn pipeline.
6. **Model choice:** **Random Forest** performed best with a **ROC-AUC of about 0.84**.
7. **App:** built a Streamlit app that turns the model into a tool anyone can use.

## Results

| Model | ROC-AUC |
|---|---|
| Logistic Regression | **~0.833** |
| **Random Forest (selected)** | **~0.844** |

## The Streamlit app

Enter a customer's details (Gender, Age, Tenure, Monthly Charge) and press **Predict**. The app shows a **risk gauge**, a **feature importance chart** and a business decision written line by line.

**Sidebar settings**

- **Churn threshold:** how risky a customer must be to count as likely to leave.
- **Retention budget:** how much you can spend on keeping customers.

**Tabs**

| Tab | What it does |
|---|---|
| Predict | Predicts churn risk for one customer |
| What-If | Changes values to see how the risk moves |
| Batch | Predicts for many customers from an uploaded file |
| Train | Retrains the model on your own labeled data |
| History | Shows earlier predictions |
| About | Explains the project |

**Discount rule used for decisions**

- Risk of **75% or more:** offer a **30–40%** discount.
- Risk below 75% but above the churn threshold: offer a **5–10%** discount.

The app has a clean, iOS-style look with light and dark mode.

## Built with

Python · pandas · NumPy · scikit-learn · imbalanced-learn · Plotly · Streamlit · Jupyter Notebook (Anaconda) · Power BI (dashboard planned)

## Project files

```
├── app.py                     # Streamlit app
├── requirements.txt           # Libraries needed
├── model files (.pkl)         # Saved Random Forest model
├── churn_analysis.ipynb       # Full analysis notebook
└── README.md
```



## How to run it on your computer

```bash
git clone [YOUR REPO LINK]
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

- The app uses only four inputs, so it is a simplified view of real customer behaviour.
- Retraining in the Train tab is not saved permanently on the free hosted version.
- Results come from one public dataset and may not match another company's customers.

## Future improvements

- Finish the **Power BI dashboard** for business users.
- Add more input features to the app.
- Try other models and tune them further.

## Build By 

- Ayush Dere


## Contact

**Ayush Dere**: [https://www.linkedin.com/in/ayush-dere-661907308] · [ayushdere96@gmail.com]
