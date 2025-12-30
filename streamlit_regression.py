import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

#Load the model
model=tf.keras.models.load_model('regressionmodel.h5')

#load scaler and encoder
with open('Data/label_encoder_gender.pkl','rb')as file:
    label_encoder_gender=pickle.load(file)

with open('Data/encode_geo.pkl','rb') as file:
    encode_geo=pickle.load(file)

with open('Data/scaler.pkl','rb') as file:
    scaler=pickle.load(file)

## streamlit app
st.title("Estimated Salary Prediction")

# User Input
geography=st.selectbox('Geography',encode_geo.categories_[0])
gender= st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balances')
credit_score=st.number_input('Credit Score')
exited=st.selectbox('Exited',[0,1])
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Num Of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "Exited": [exited],
})

## one hot encoded geography
geo = encode_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo,columns=encode_geo.get_feature_names_out(['Geography']))

#combine encoded column with input data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#scaled dta
input_data_scaled = scaler.transform(input_data)

#prediction churn
prediction=model.predict(input_data_scaled)

prediction_salary=prediction[0][0]

st.write(f'Predicted Salary: {prediction_salary:.2f}')

