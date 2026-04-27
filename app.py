import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

st.set_page_config(
    page_title="Diabetes Prediction App | ML Project",
)
# -------------------------------
# LOAD DATA & MODEL
# -------------------------------

@st.cache_data
def load_data():
    return pd.read_csv('data/diabetes.csv')

@st.cache_resource
def load_model():
    return pickle.load(open('model/model.pkl', 'rb'))

df = load_data()
rf_model = load_model()

# Train logistic regression for comparison
@st.cache_resource
def train_logistic():
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

lr_model = train_logistic()

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(page_title="Diabetes AI Dashboard", layout="wide")

st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>🧠 Diabetes AI Dashboard</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# USER INPUT
# -------------------------------

st.sidebar.header("🧾 Patient Input")

def user_input():
    return pd.DataFrame({
        'Pregnancies': [st.sidebar.slider('Pregnancies', 0, 17, 3)],
        'Glucose': [st.sidebar.slider('Glucose', 0, 200, 120)],
        'BloodPressure': [st.sidebar.slider('Blood Pressure', 0, 122, 70)],
        'SkinThickness': [st.sidebar.slider('Skin Thickness', 0, 100, 20)],
        'Insulin': [st.sidebar.slider('Insulin', 0, 846, 79)],
        'BMI': [st.sidebar.slider('BMI', 0.0, 67.0, 20.0)],
        'DiabetesPedigreeFunction': [st.sidebar.slider('DPF', 0.0, 2.4, 0.47)],
        'Age': [st.sidebar.slider('Age', 21, 88, 33)]
    })

user_data = user_input()

# -------------------------------
# PREDICTIONS
# -------------------------------

rf_pred = rf_model.predict(user_data)[0]
lr_pred = lr_model.predict(user_data)[0]

# -------------------------------
# MAIN DISPLAY
# -------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧾 Patient Data")
    st.write(user_data)

with col2:
    st.subheader("🧠 Prediction Results")

    st.write("**Random Forest:**")
    st.success("Low Risk") if rf_pred == 0 else st.error("High Risk")

    st.write("**Logistic Regression:**")
    st.success("Low Risk") if lr_pred == 0 else st.error("High Risk")

# -------------------------------
# METRICS
# -------------------------------

st.markdown("### 📈 Key Metrics")

m1, m2, m3 = st.columns(3)
user = user_data.iloc[0]

m1.metric("Glucose", int(user["Glucose"]))
m2.metric("BMI", float(user["BMI"]))
m3.metric("Age", int(user["Age"]))

# -------------------------------
# FEATURE IMPORTANCE (RF)
# -------------------------------

st.markdown("## 📊 Feature Importance (Random Forest)")

importances = rf_model.feature_importances_
features = df.drop('Outcome', axis=1).columns

imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

fig = plt.figure()
sns.barplot(x='Importance', y='Feature', data=imp_df)
st.pyplot(fig)

# -------------------------------
# ROC CURVE
# -------------------------------

st.markdown("## 📉 ROC Curve Comparison")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

rf_probs = rf_model.predict_proba(X)[:,1]
lr_probs = lr_model.predict_proba(X)[:,1]

rf_fpr, rf_tpr, _ = roc_curve(y, rf_probs)
lr_fpr, lr_tpr, _ = roc_curve(y, lr_probs)

rf_auc = auc(rf_fpr, rf_tpr)
lr_auc = auc(lr_fpr, lr_tpr)

fig2 = plt.figure()
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC={rf_auc:.2f})")
plt.plot(lr_fpr, lr_tpr, label=f"Logistic (AUC={lr_auc:.2f})")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
st.pyplot(fig2)

# -------------------------------
# VISUALIZATION
# -------------------------------

st.markdown("## 📊 Patient vs Population")

def plot(feature):
    fig = plt.figure()
    sns.scatterplot(x='Age', y=feature, data=df, hue='Outcome')
    sns.scatterplot(
        x=user_data['Age'],
        y=user_data[feature],
        s=200,
        color='red' if rf_pred else 'blue'
    )
    st.pyplot(fig)

for f in ['Glucose', 'BMI', 'Insulin']:
    st.subheader(f"{f} vs Age")
    plot(f)

# -------------------------------
# DOWNLOAD REPORT
# -------------------------------

st.markdown("## 📄 Download Report")

report = user_data.copy()
report["RF_Prediction"] = rf_pred
report["LR_Prediction"] = lr_pred

st.download_button(
    label="Download Report (CSV)",
    data=report.to_csv(index=False),
    file_name="diabetes_report.csv",
    mime="text/csv"
)

# -------------------------------
# SIDEBAR INFO
# -------------------------------

st.sidebar.markdown("## ℹ️ About")
st.sidebar.write("ML-powered Diabetes Risk Prediction Dashboard")