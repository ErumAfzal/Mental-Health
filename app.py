import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Mental Health Stress Level Predictor (Bhutan)")

MODEL_PATH = "mental_health_model.pkl"

# If model not found, train a fresh one
if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Training a new model...")

    import numpy as np
    np.random.seed(42)
    size = 600

    data = pd.DataFrame({
        'age': np.random.randint(15, 60, size),
        'sleep_hours': np.random.uniform(4, 10, size),
        'social_interaction': np.random.randint(0, 7, size),
        'work_stress': np.random.randint(1, 10, size),
        'physical_activity': np.random.randint(0, 6, size),
        'mood_score': np.random.randint(1, 10, size)
    })

    score = (data['work_stress'] * 0.5) + (10 - data['mood_score']) + (6 - data['physical_activity'])
    conditions = [score < 8, score < 14, score >= 14]
    choices = ['low', 'medium', 'high']
    data['stress_level'] = np.select(conditions, choices)

    X = data.drop('stress_level', axis=1)
    y = data['stress_level']

    model = RandomForestClassifier()
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    st.success("Model trained successfully.")

else:
    model = pickle.load(open(MODEL_PATH, "rb"))

age = st.slider('Age', 15, 60, 25)
sleep_hours = st.slider('Sleep Hours per Night', 4.0, 10.0, 7.0)
social_interaction = st.slider('Days per Week with Social Interaction', 0, 7, 3)
work_stress = st.slider('Work/Study Stress Level (1-10)', 1, 10, 5)
physical_activity = st.slider('Physical Activity Days per Week', 0, 6, 2)
mood_score = st.slider('Mood Score (1=low, 10=high)', 1, 10, 6)

if st.button("Predict Stress Level"):
    features = np.array([[age, sleep_hours, social_interaction, work_stress, physical_activity, mood_score]])
    prediction = model.predict(features)[0]
    st.write("Predicted Stress Level:", prediction)
