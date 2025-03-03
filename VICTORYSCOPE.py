# Importing the libraries 
import streamlit as st 
import pickle as pkl 
import pandas as pd 

# Setting page layout
st.set_page_config(layout="wide")

# Title of the page
st.title("VICTORYSCOPE - IPL Win Predictor")

# Importing data and model from pickle
teams = pkl.load(open('team.pkl','rb'))
cities = pkl.load(open('city.pkl','rb'))
model = pkl.load(open('model.pkl','rb'))

# First row of input columns
col1, col2, col3 = st.columns(3)
with col1: 
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2: 
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))
with col3: 
    selected_city = st.selectbox('Select the host city', sorted(cities))

# Input for target score
target = st.number_input('Target Score', min_value=0, max_value=720, step=1)

# Second row of input columns
col4, col5, col6 = st.columns(3)
with col4:
    score = st.number_input('Score', min_value=0, max_value=720, step=1)
with col5:
    overs = st.number_input('Overs Done', min_value=0.1, max_value=20.0, step=0.1)  # Allow float values
with col6: 
    wickets_fell = st.number_input('Wickets Fell', min_value=0, max_value=10, step=1)

# When the user clicks the 'Predict Probabilities' button
if st.button('Predict Probabilities'):
    try:
        # Derived features
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_remaining = 10 - wickets_fell
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Creating input DataFrame
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

        # Ensure the data structure is compatible with the pipeline
        transformed_input = model.steps[0][1].transform(input_df)  # Preprocessing step
        result = model.steps[-1][1].predict_proba(transformed_input)  # Prediction step

        # Extract probabilities
        loss = result[0][0]
        win = result[0][1]

        # Display predictions
        st.header(f"🏏 {batting_team} Winning Probability: {round(win * 100)}%")
        st.header(f"🏏 {bowling_team} Winning Probability: {round(loss * 100)}%")
    
    except Exception as e:
        st.error(f"⚠️ Error in model processing: {e}")
