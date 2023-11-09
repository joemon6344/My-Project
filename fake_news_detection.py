# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 22:17:40 2023

@author: admin
"""
import streamlit as st
import pickle
import numpy as np  # Import numpy for data processing

# Load the pickled model
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit app with styling and emojis
def main():
    st.title('🕵‍♂ Fake News Detector')

    st.write(
        "Welcome to the Fake News Detector app! Enter a news snippet below and click the 'Check' button to determine whether it's fake or real news."
    )

    st.markdown('---')

    # User input for text
    user_input = st.text_area('Enter the news text:', height=150)

    if st.button('Check'):
        # Make a prediction using the loaded model
        user_input_processed = preprocess_text(user_input)  # You need to define this function
        prediction = model.predict([user_input_processed])
        result = 'Fake News' if prediction[0] == 1 else 'Real News'
        
        # Display result with emoji and styling
        result_emoji = "❌" if prediction[0] == 1 else "✅"
        st.markdown(f'## Result: {result_emoji} {result}', unsafe_allow_html=True)

    st.markdown('---')

def preprocess_text(text):
    # Define your text preprocessing steps here (tokenization, stemming, etc.)
    # You'll need to process the input text before making predictions
    processed_text = text  # Placeholder for preprocessing
    return processed_text

if __name__ == '__main__':
    main()

