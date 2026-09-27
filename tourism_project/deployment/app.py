import streamlit as st
import pandas as pd
import joblib
import os

# Load the model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'best_model.joblib')
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.stop()
    model = joblib.load(model_path)
    return model

model = load_model()

st.title('Wellness Tourism Package Prediction')
st.write('Enter customer details to predict if they will purchase the Wellness Tourism Package.')

# Input fields
occupation_options = ['Salaried', 'Small Business', 'Free Lancer', 'Large Business', 'Government Sector']
gender_options = ['Male', 'Female']
marital_status_options = ['Married', 'Single', 'Divorced']
type_of_contact_options = ['Self Inquiry', 'Company Invited']
product_pitched_options = ['Resort', 'Basic', 'Standard', 'Deluxe', 'Super Deluxe']
designation_options = ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP']

with st.form('prediction_form'):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider('Age', 18, 100, 30)
        typeof_contact = st.selectbox('Type of Contact', type_of_contact_options)
        city_tier = st.selectbox('City Tier', [1, 2, 3])
        occupation = st.selectbox('Occupation', occupation_options)
        gender = st.selectbox('Gender', gender_options)
        number_of_person_visiting = st.number_input('Number of Persons Visiting', 1, 10, 1)
        preferred_property_star = st.selectbox('Preferred Property Star', [3, 4, 5])
        marital_status = st.selectbox('Marital Status', marital_status_options)
        number_of_trips = st.number_input('Number of Trips Annually', 0, 20, 1)

    with col2:
        passport = st.selectbox('Passport', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        own_car = st.selectbox('Own Car', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        number_of_children_visiting = st.number_input('Number of Children Visiting', 0, 5, 0)
        designation = st.selectbox('Designation', designation_options)
        monthly_income = st.number_input('Monthly Income', 10000.0, 100000.0, 25000.0, step=1000.0)
        pitch_satisfaction_score = st.slider('Pitch Satisfaction Score', 1, 5, 3)
        product_pitched = st.selectbox('Product Pitched', product_pitched_options)
        number_of_followups = st.number_input('Number of Followups', 0, 10, 3)
        duration_of_pitch = st.number_input('Duration of Pitch (minutes)', 5.0, 60.0, 15.0, step=1.0)

    submitted = st.form_submit_button('Predict')

    if submitted:
        # Create a DataFrame from inputs
        input_data = pd.DataFrame({
            'Age': [age],
            'TypeofContact': [typeof_contact],
            'CityTier': [city_tier],
            'Occupation': [occupation],
            'Gender': [gender],
            'NumberOfPersonVisiting': [number_of_person_visiting],
            'PreferredPropertyStar': [preferred_property_star],
            'MaritalStatus': [marital_status],
            'NumberOfTrips': [number_of_trips],
            'Passport': [passport],
            'OwnCar': [own_car],
            'NumberOfChildrenVisiting': [number_of_children_visiting],
            'Designation': [designation],
            'MonthlyIncome': [monthly_income],
            'PitchSatisfactionScore': [pitch_satisfaction_score],
            'ProductPitched': [product_pitched],
            'NumberOfFollowups': [number_of_followups],
            'DurationOfPitch': [duration_of_pitch]
        })

        # Predict
        try:
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]

            st.subheader('Prediction Result:')
            if prediction == 1:
                st.success(f'The customer is likely to purchase the package (Probability: {prediction_proba[1]:.2f})')
            else:
                st.error(f'The customer is unlikely to purchase the package (Probability: {prediction_proba[0]:.2f})')

            st.write("--- Input Data for Prediction ---")
            st.dataframe(input_data)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Please check the input data and ensure all necessary features are provided in the correct format.")
