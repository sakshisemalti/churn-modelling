# 📊 Customer Churn Prediction App

This project is a **Streamlit web application** that predicts whether a customer is likely to churn based on their profile and account details. It uses a trained **TensorFlow/Keras model** along with **scikit‑learn preprocessing (StandardScaler, LabelEncoder, OneHotEncoder)**.

---

## 🚀 Features
- Interactive **Streamlit UI** for user input  
- Encodes categorical variables (`Geography`, `Gender`) using saved encoders  
- Scales numeric features with a saved `StandardScaler`  
- Predicts churn probability using a trained deep learning model (`model.h5`)  
- Displays probability and churn decision directly in the app  

---

## 🛠 Tech Stack
- **Python 3.12+**
- **TensorFlow / Keras** – deep learning model
- **scikit‑learn** – preprocessing (scaler, encoders)
- **Streamlit** – web app framework
- **Pickle** – saving/loading encoders and scaler

---

## 📂 Project Structure
├── app.py                          # Streamlit app for interactive churn prediction
├── model.h5                       # Trained Keras model
├── Data/
│   ├── label_encoder_gender.pkl   # Saved LabelEncoder for Gender
│   ├── encode_geo.pkl             # Saved OneHotEncoder for Geography
│   └── scaler.pkl                  # Saved StandardScaler
├── training.py                     # Script used to train the churn model
├── prediction_test_example.ipynb  # Notebook with example predictions on test data
├── prediction.ipynb                # Notebook for running predictions interactively
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/sakshisemalti/churn-modelling.git
   cd churn-modelling
   
2. **Create a virtual environment (recommended)
   ```bash
    python -m venv venv
    source venv/bin/activate   # Mac/Linux
   
3. **Install dependencies
   ```bash
    pip install -r requirements.txt

4. **Run the Streamlit app
   ```bash
    streamlit run app.py

