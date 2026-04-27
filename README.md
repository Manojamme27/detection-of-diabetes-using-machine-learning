# 🧠 Diabetes Prediction AI Dashboard

🚀 A full-stack Machine Learning web application that predicts diabetes risk using multiple models and presents results through an interactive dashboard.

🔗 **Live App:** https://your-app.streamlit.app  

---

## 📌 Overview

This project combines **Machine Learning + Data Visualization + Web UI** to build a real-world healthcare prediction system.

Users can:
- Input patient medical data
- Get predictions from multiple ML models
- Analyze feature importance
- Visualize comparisons with real dataset
- Download prediction reports

---

## ✨ Features

### 🧠 Machine Learning
- Random Forest Classifier (Primary Model)
- Logistic Regression (Model Comparison)
- ROC Curve & AUC Evaluation

### 📊 Data Insights
- Feature Importance visualization
- Patient vs Population comparison graphs
- Dataset statistical summary

### 🖥️ Interactive Dashboard
- Clean UI built with Streamlit
- Real-time predictions
- Key metrics display (Glucose, BMI, Age)
- Sidebar input controls

### 📄 Export
- Download prediction report as CSV

---

## 🛠️ Tech Stack

### 🔹 Machine Learning
- Python
- Scikit-learn

### 🔹 Data Processing
- Pandas
- NumPy

### 🔹 Visualization
- Matplotlib
- Seaborn

### 🔹 Frontend / UI
- Streamlit

---

## 📁 Project Structure
```bash
detection-of-diabetes-using-machine-learning/
│── data/
│ └── diabetes.csv
│
│── model/
│ ├── model.pkl
│ └── train_model.py
│
│── utils/
│ └── helpers.py
│
│── app.py
│── requirements.txt
│── runtime.txt
│── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/detection-of-diabetes-using-machine-learning.git
cd detection-of-diabetes-using-machine-learning
```

## Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
```
## Install dependencies
```bash
pip install -r requirements.txt
```
## Train the model
```bash
cd model
python train_model.py
cd ..
```
## Run the app
```bash
streamlit run app.py
```
## 📊 Model Details
- Dataset: PIMA Indians Diabetes Dataset
- Target Variable: Outcome
- Models Used:
  - Random Forest Classifier
  - Logistic Regression
- Evaluation:
  - ROC Curve
  - AUC Score
## 📷 Screenshots
<img width="1440" height="815" alt="Screenshot 2026-04-27 at 11 20 23 PM" src="https://github.com/user-attachments/assets/2e445695-a490-408d-8bfd-3648d9d7a7fb" />
<img width="1440" height="815" alt="Screenshot 2026-04-27 at 11 20 48 PM" src="https://github.com/user-attachments/assets/a6c77278-5780-4b0c-b89c-1b13c4e29a40" />
<img width="1440" height="815" alt="Screenshot 2026-04-27 at 11 20 43 PM" src="https://github.com/user-attachments/assets/0cf050a3-8343-41a6-8cce-47e93e325c83" />
<img width="1440" height="815" alt="Screenshot 2026-04-27 at 11 20 33 PM" src="https://github.com/user-attachments/assets/04f3ff54-2ebd-4dad-aa4a-236d82c65a1f" />
<img width="1440" height="815" alt="Screenshot 2026-04-27 at 11 21 00 PM" src="https://github.com/user-attachments/assets/90452c2c-4a31-4852-9f2b-b85b90c9f4c3" />
<img width="1440" height="815" alt="Screenshot 2026-04-27 at 11 21 13 PM" src="https://github.com/user-attachments/assets/559ed453-c526-4452-af58-90cb4a28bc4d" />
<img width="1440" height="815" alt="Screenshot 2026-04-27 at 11 21 19 PM" src="https://github.com/user-attachments/assets/7468f199-08b0-433f-a137-88d5520c7dbe" />
<img width="1440" height="815" alt="Screenshot 2026-04-27 at 11 21 25 PM" src="https://github.com/user-attachments/assets/cb7d8b23-f371-47ff-9b23-ccdbabeb11b6" />
<img width="1440" height="815" alt="Screenshot 2026-04-27 at 11 21 06 PM" src="https://github.com/user-attachments/assets/19574218-a14a-451a-a32c-2f5b27c87c09" />
<img width="1440" height="815" alt="Screenshot 2026-04-27 at 11 21 30 PM" src="https://github.com/user-attachments/assets/b6a37def-58e2-4105-8201-3779de0c1956" />

## 🎯 Sample Prediction
```bash
Input:

Glucose: 120
BMI: 28
Age: 35

Output:
➡️ Low Risk
```
## 💡 Why This Project?

This project demonstrates:

- End-to-end ML pipeline
- Model comparison & evaluation
- Interactive data visualization
- Real-world deployment

## 🚀 Future Improvements
- 🔐 User authentication
- 💳 Integration with healthcare APIs
- 📱 Mobile optimization
- ☁️ Scalable backend (FastAPI)
## 📫 Contact
- GitHub: https://github.com/Manojamme27
- LinkedIn: https://www.linkedin.com/in/amme-manoj-4569b1228/
## 📜 License

MIT License

