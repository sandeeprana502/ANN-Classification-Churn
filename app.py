import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard


# Load the trained model
model = load_model("Model.h5")
# Load the pickle file
with open("label_encoder_gender.pkl","rb") as file:
    label_encoder_gen = pickle.load(file)
with open("geo_encoder_location.pkl","rb") as file:
    geo_encoder_location = pickle.load(file)
with open("scalar_dataset.pkl","rb") as file:
    scaler = pickle.load(file)

# Steamlit app
st.title("Churn Prediction Model")

#user input
Geography = st.selectbox('Geography',geo_encoder_location.categories_[0])
gender = st.selectbox('Gender',label_encoder_gen.classes_)
age = st.slider('Age',18,90)
balance = st.number_input('Balance')
credit_score = st.number_input('CreditScore')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,13)
number_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])


# prepare the input
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [Geography],
    'Gender':[label_encoder_gen.transform([gender])[0]],
    'Age':[age],
    'Tenure': [tenure],
    'Balance':[balance],
    'NumOfProducts':[number_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary],
})

# One hot encoding
geo_encoded = geo_encoder_location.transform([[Geography]]).toarray()
# geo_encoded = geo_encoder_location.transform([input_data['Geography']])
geo_location = pd.DataFrame(geo_encoded,columns=geo_encoder_location.get_feature_names_out(['Geography']))
# st.write(geo_location)
input_data = input_data.drop(['Geography'],axis =1)
# st.write(input_data)
#Combining
input_data = pd.concat([input_data.reset_index(drop=True),geo_location],axis=1)
# st.write(input_data)
#Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict Churn
prediction = model.predict(input_data_scaled)


if prediction[0][0] > 0.50:
    st.write('Model has predicted churn with accuracy ' + str(prediction[0][0]))
else:
    st.write('Model has predicted no churn with accuracy '+ str(prediction[0][0]))

