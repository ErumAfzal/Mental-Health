import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option("deprecation.showPyplotGlobalUse", False)

st.title("Mental Health Stress Level Predictor with Visualizations (Bhutan)")

# --------------------------
# DATA & MODEL PREPARATION
# --------------------------

@st.cache_resource
def load_data():
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

    conditions = [
        (score < 8),
        ((score >= 8) & (score < 14)),
        (score >= 14)
    ]
    choices = ['low', 'medium', 'high']

    data['stress_level'] = np.select(conditions, choices, default='medium')

    return data


@st.cache_resource
def train_model(data):
    X = data.drop("stress_level", axis=1)
    y = data["stress_level"]

    model = RandomForestClassifier()
    model.fit(X, y)

    return model


data = load_data()
model = train_model(data)

# --------------------------
# SIDEBAR NAVIGATION
# --------------------------

menu = st.sidebar.selectbox(
    "Navigate",
    ["Dataset Overview", "Visualizations", "Train Model Summary", "Predict Stress Level"]
)

# --------------------------
# DATASET OVERVIEW SECTION
# --------------------------

if menu == "Dataset Overview":
    st.header("Dataset Overview")

    st.write("Preview the first rows of the dataset:")
    st.dataframe(data.head())

    st.write("Basic statistics:")
    st.dataframe(data.describe())

    st.write("Stress Level Distribution:")
    st.bar_chart(data["stress_level"].value_counts())

# --------------------------
# VISUALIZATION SECTION
# --------------------------

elif menu == "Visualizations":
    st.header("Visualizations")

    st.write("Select visualization type:")
    viz_type = st.selectbox(
        "Choose a chart type:",
        [
            "Correlation Heatmap",
            "Line Chart",
            "Bar Chart",
            "Area Chart",
            "Histogram",
            "Scatter Plot"
        ]
    )

    # Correlation Heatmap
    if viz_type == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        corr = data.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        st.pyplot()

    # Line Chart
    elif viz_type == "Line Chart":
        st.subheader("Line Chart")
        st.line_chart(data.drop("stress_level", axis=1))

    # Bar Chart
    elif viz_type == "Bar Chart":
        st.subheader("Bar Chart")

        feature = st.selectbox("Select a feature:", data.columns[:-1])
        st.bar_chart(data[feature])

    # Area Chart
    elif viz_type == "Area Chart":
        st.subheader("Area Chart")
        st.area_chart(data.drop("stress_level", axis=1))

    # Histogram
    elif viz_type == "Histogram":
        feature = st.selectbox("Select a numeric feature:", data.columns[:-1])

        plt.hist(data[feature], bins=20)
        st.pyplot()

    # Scatter Plot
    elif viz_type == "Scatter Plot":
        x_axis = st.selectbox("X-axis:", data.columns[:-1])
        y_axis = st.selectbox("Y-axis:", data.columns[:-1])

        plt.scatter(data[x_axis], data[y_axis])
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        st.pyplot()

# --------------------------
# MODEL SUMMARY SECTION
# --------------------------

elif menu == "Train Model Summary":
    st.header("Model Training Summary")

    st.write("The model used is a Random Forest Classifier.")

    X = data.drop("stress_level", axis=1)
    feature_importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    st.write("Feature Importance Chart:")
    st.bar_chart(importance_df.set_index("Feature"))

    st.write("Feature Importance (Table):")
    st.dataframe(importance_df)

# --------------------------
# PREDICTION SECTION
# --------------------------

elif menu == "Predict Stress Level":
    st.header("Predict Mental Health Stress Level")

    age = st.slider('Age', 15, 60, 25)
    sleep_hours = st.slider('Sleep Hours per Night', 4.0, 10.0, 7.0)
    social_interaction = st.slider('Social Interaction (days/week)', 0, 7, 3)
    work_stress = st.slider('Work or Study Stress Level (1-10)', 1, 10, 5)
    physical_activity = st.slider('Physical Activity (days/week)', 0, 6, 2)
    mood_score = st.slider('Mood Score (1 low, 10 high)', 1, 10, 6)

    if st.button("Predict"):
        features = np.array([[age, sleep_hours, social_interaction, work_stress, physical_activity, mood_score]])
        prediction = model.predict(features)[0]
        st.write("Predicted Stress Level:", prediction)
