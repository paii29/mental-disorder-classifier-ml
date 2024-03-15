import streamlit as st
import pickle
from sklearn.linear_model import LogisticRegression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


occurance = ['Usually', 'Sometimes', 'Most-Often', 'Seldom']
yes_no = ['No', 'Yes']

mhds = pd.read_csv('Dataset-Mental-Disorders.csv')
mhds = mhds.drop('Patient Number', axis =1)


def show_page() :
    st.title('What is your mental state?')
    with st.form('my_form'):
        

        X = mhds.drop('Expert Diagnose',axis=1) #features
        y = mhds['Expert Diagnose'] #target 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

        encoder = OneHotEncoder(handle_unknown='ignore')
        X_train_encoded = encoder.fit_transform(X_train)
        X_test_encoded = encoder.transform(X_test)

        logreg = LogisticRegression()
        logreg.fit(X_train_encoded, y_train)
        
        
        sadness = st.selectbox('How often do you feel sadness?', occurance)
        euphoric = st.selectbox('How often do you feel euphoric/happy?', occurance)
        exhausted = st.selectbox('How often do you feel exhausted?', occurance)
        sleep_dissorder = st.selectbox('How often do you encounter insomnia?', occurance)
        #mood_swing = st.toggle('Mood Swing')
        mood_swing = st.selectbox('Do you encounter mood swing?', yes_no)
        suicidal_thoughts = st.selectbox('Do you get suicidal thoughts?', yes_no)
        anorxia = st.selectbox('Do you have eating disorder?', yes_no)
        authority_respect = st.selectbox('Do you have authority or respect to others?', yes_no)
        try_explanation = st.selectbox('Do you try and explain out of anxiety?', yes_no)
        aggressive_response = st.selectbox('Do you have aggressive reaction?', yes_no)
        ignore_moveon = st.selectbox('Can you easily ignore and move on?', yes_no)
        nervous_breakdown = st.selectbox('Do you encounter nervous break-down?', yes_no)
        admit_mistakes = st.selectbox('Do you admit mistakes?', yes_no)
        overthinking = st.selectbox('Do you overthink?', yes_no)
        sexual_activity = st.slider('Rate your sexual desire', 0, 10, 0, 1)
        sexual_activity = str(sexual_activity) + ' From 10'
        concentration = st.slider('Rate your concentration', 0, 10, 0, 1)
        concentration = str(concentration) + ' From 10'
        optimisim = st.slider('Rate your optimisim', 0, 10, 0, 1)
        optimisim = str(optimisim) + ' From 10'

        submitted = st.form_submit_button("Submit")
        if submitted:
            inputData ={'Sadness':sadness,'Euphoric':euphoric, 'Exhausted':exhausted,'Sleep dissorder':sleep_dissorder ,'Mood Swing':mood_swing ,'Suicidal thoughts':suicidal_thoughts ,'Anorxia':anorxia ,'Authority Respect':authority_respect ,'Try-Explanation':try_explanation ,'Aggressive Response':aggressive_response ,'Ignore & Move-On':ignore_moveon ,'Nervous Break-down':nervous_breakdown ,'Admit Mistakes':admit_mistakes ,'Overthinking':overthinking ,'Sexual Activity':sexual_activity ,'Concentration':concentration ,'Optimisim':optimisim }

            features = pd.DataFrame(inputData, index=[0])
            features = encoder.transform(features)

            y_preds = logreg.predict(features)
            st.write("Predicted mental state is", y_preds[0] ,".") 

show_page()
