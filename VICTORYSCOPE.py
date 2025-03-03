# Importing necessary libraries
import streamlit as st 
import pickle as pkl 
import pandas as pd 

# Set page layout
st.set_page_config(layout="wide")

# Title of the app
st.title("VICTORYSCOPE - IPL Win Predictor")

# Load pre-trained model and supporting files
teams = pkl.load(open('team.pkl','rb'))
cities = pkl.load(open('city.pkl','rb'))
model = pkl.load(open('model.pkl','rb'))  # Ensure this is a pipeline with preprocessing steps

# First row for team selections
col1, col2, col3 = st.columns(3)
with col1: 
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2: 
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))
with col3: 
    selected_city = st.selectbox('Select the host city', sorted(cities))

# Input for match conditions
target = st.number_input('Target Score', min_value=0, max_value=720, step=1)

col4, col5, col6 = st.columns(3)
with col4:
    score = st.number_input('Score', min_value=0, max_value=720, step=1)
with col5:
    overs = st.number_input('Overs Done', min_value=0.1, max_value=20.0, step=0.1)  # Allow fractional overs
with col6: 
    wickets_fell = st.number_input('Wickets Fell', min_value=0, max_value=10, step=1)

# Prediction button
if st.button('Predict Probabilities'):
    try:
        # Derived Features
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_remaining = 10 - wickets_fell
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Creating DataFrame with exact feature names used during model training
        input_df = pd.DataFrame({
            'batting_team': [batting_team], 
            'bowling_team': [bowling_team], 
            'city': [selected_city], 
            'Score': [score],
            'Wickets': [wickets_remaining],
            'Remaining Balls': [balls_left], 
            'target_left': [runs_left], 
            'crr': [crr], 
            'rrr': [rrr]
        })

        # Ensure input_df is processed correctly by the model pipeline
        transformed_input = model.named_steps['preprocessor'].transform(input_df)  # Replace 'preprocessor' with actual step name
        result = model.named_steps['classifier'].predict_proba(transformed_input)  # Replace 'classifier' with actual step name

        # Extract probabilities
        loss = result[0][0]
        win = result[0][1]

        # Display predictions
        st.header(f"🏏 {batting_team} Winning Probability: {round(win * 100)}%")
        st.header(f"🏏 {bowling_team} Winning Probability: {round(loss * 100)}%")

    except Exception as e:
        st.error(f"⚠️ Error in model processing: {e}")
