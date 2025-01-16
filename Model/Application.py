import streamlit as st
import pandas as pd
import joblib
import os
st.write(os.path.exists('./Model/model_trend.pkl'))  # Should return True if the file exists


model_trend = joblib.load('./Model/model_trend.pkl')
scaler = joblib.load('./Model/scaler/scaler.pkl')

label_encoders = {
    'Gender': joblib.load('./Model/encoder/label_encoder_gender.pkl'),
    'Category': joblib.load('./Model/encoder/label_encoder_category.pkl'),
    'State': joblib.load('./Model/encoder/label_encoder_state.pkl'),
    'Season': joblib.load('./Model/encoder/label_encoder_season.pkl'),
    'Item Purchased': joblib.load('./Model/encoder/label_encoder_item purchased.pkl')
}
def predict_trend_item(age, gender, category, state, season):
    new_customer_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Category': [category],
        'State': [state],
        'Season': [season]
    })
    
    # Transform categorical features
    for col in ['Gender', 'Category', 'State', 'Season']:
        new_customer_data[col] = label_encoders[col].transform(new_customer_data[col])
    
    # Scale the features
    new_customer_data_scaled = scaler.transform(new_customer_data[['Age', 'Gender', 'Category', 'State', 'Season']])
    
    # Predict the item
    predicted_item = model_trend.predict(new_customer_data_scaled)
    predicted_item_name = label_encoders['Item Purchased'].inverse_transform(predicted_item)
    return predicted_item_name[0]

# Streamlit UI
st.title('Product Trend Prediction')

# User input fields
age = st.slider('Age', min_value=18, max_value=70, value=25)
gender = st.selectbox('Gender', options=label_encoders['Gender'].classes_)
category = st.selectbox('Category', options=label_encoders['Category'].classes_)
state = st.selectbox('State', options=label_encoders['State'].classes_)
season = st.selectbox('Season', options=label_encoders['Season'].classes_)

if st.button('Predict'):
    predicted_item = predict_trend_item(age, gender, category, state, season)
    if predicted_item:
        st.success(f'Predicted Product for the Customer: {predicted_item}')