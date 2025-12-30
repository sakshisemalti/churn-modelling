# ANN Churn Prediction & Salary Regression

- **Churn Prediction App** → [Live Demo](https://ann-classification-churn-nlpcham.streamlit.app/)- predicts whether a customer is likely to churn based on their profile.  
- **Salary Prediction App** → [Live Demo](https://salary-prediction-ann.streamlit.app/) — predicts salary using a regression model.

---

## Features

### Churn Prediction
- Interactive **Streamlit UI** for customer profile input  
- Encodes categorical variables (`Geography`, `Gender`) using saved encoders  
- Scales numeric features with a saved `StandardScaler`  
- Predicts churn probability using a trained deep learning model (`model.h5`)  
- Displays probability and churn decision directly in the app  

### Salary Regression
- Separate **Streamlit app** for salary prediction (`streamlit_salary.py`)  
- Uses a trained regression ANN model (`salaryregressionmodel.h5`)  
- Logs training runs in `regressionlogs/` for reproducibility  
- Provides interactive salary prediction based on input features  

---

## Tech Stack
- **Python 3.12+**
- **TensorFlow / Keras** – deep learning models (classification & regression)
- **scikit‑learn** – preprocessing (scaler, encoders)
- **Streamlit** – web app framework
- **Pickle** – saving/loading encoders and scaler
- **Jupyter Notebook** – for experiments and prediction examples

---

## Project Structure
```
├── app.py                         # Streamlit app for churn prediction
├── model.h5                       # Trained Keras churn model
├── salaryregressionmodel.h5       # New trained Keras regression model (salary prediction)
├── streamlit_regression.py        # Script to train/run regression model
├── streamlit_salary.py            # Streamlit app for salary regression predictions
├── regressionlogs/
│   └── fit31/                     # Logs directory for regression training and validation runs
├── Data/
│   ├── label_encoder_gender.pkl   # Saved LabelEncoder for Gender (churn model)
│   ├── encode_geo.pkl             # Saved OneHotEncoder for Geography (churn model)
│   └── scaler.pkl                 # Saved StandardScaler (churn model)
├── training.py                    # Script used to train churn model
├── prediction_test_example.ipynb  # Notebook with churn test predictions
├── prediction.ipynb               # Notebook for churn interactive predictions
├── requirements.txt               # Python dependencies
└── README.md                      # Documentation
```
---

## Setup Instructions

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

4. **Run the Churn Prediction app
   ```bash
    streamlit run app.py
   
5. **Run the Salary Regression app
   ```bash
   streamlit run streamlit_salary.py

