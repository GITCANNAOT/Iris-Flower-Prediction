import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("iris_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Sidebar section
st.sidebar.title("⚙️ Menu")

# Toggle for About Section
show_about = st.sidebar.toggle("ℹ️ About This Website")

if show_about:
    st.sidebar.markdown("""
### 🌸 Iris Flower Prediction App

This web application predicts the **species of an Iris flower** using a Machine Learning model.

### 🤖 How it Works
- The model is trained using the **Iris dataset**.
- Users input flower measurements:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- The ML model analyzes these features and predicts the species.

### 🌼 Possible Predictions
- **Iris Setosa**
- **Iris Versicolor**
- **Iris Virginica**

### 🛠 Technologies Used
- Python
- Streamlit
- Scikit-learn
- NumPy & Pandas
- Matplotlib

### 🚀 Purpose
This project demonstrates how **Machine Learning models can be deployed as interactive web applications**.
""")
# -----------------------------
# Dark / Light Mode Toggle
# -----------------------------
mode = st.sidebar.toggle("🌙 Dark Mode")

if mode:
    background = "#0E1117"
    text = "white"
else:
    background = "#F5F7FA"
    text = "black"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {background};
        color: {text};
    }}

    .title {{
        font-size:50px;
        font-weight:bold;
        text-align:center;
        background: -webkit-linear-gradient(#ff4b4b,#ff9f43);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .card {{
        background: rgba(255,255,255,0.15);
        padding:20px;
        border-radius:20px;
        box-shadow:0 8px 32px rgba(0,0,0,0.2);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Header
# -----------------------------
st.markdown('<p class="title">🌸 Iris Flower Prediction</p>', unsafe_allow_html=True)

st.markdown(
"""
### Predict the species of an Iris flower using Machine Learning

Enter flower measurements and see the AI prediction instantly.
"""
)

# -----------------------------
# Layout Columns
# -----------------------------
col1, col2 = st.columns([1,1])


# -----------------------------
# Input Section
# -----------------------------
with col1:
    st.markdown("## 🌿 Flower Measurements")

    sepal_length = st.slider("Sepal Length",4.0,8.0,5.4)
    sepal_width = st.slider("Sepal Width",2.0,4.5,3.4)
    petal_length = st.slider("Petal Length",1.0,7.0,1.3)
    petal_width = st.slider("Petal Width",0.1,2.5,0.2)

    predict = st.button("🔍 Predict Flower")


# -----------------------------
# Prediction Section
# -----------------------------
with col2:

    if predict:

        features = np.array([[sepal_length,sepal_width,petal_length,petal_width]])

        prediction = model.predict(features)
        proba = model.predict_proba(features)

        species = encoder.inverse_transform(prediction)[0]

        st.success(f"🌸 Predicted Flower: **{species}**")

        # Flower Image
        if species == "setosa":
            st.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg")

        elif species == "versicolor":
            st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg")

        else:
            st.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg")

        # Probability Graph
        st.subheader("📊 Prediction Probability")

        fig, ax = plt.subplots()
        labels = encoder.classes_
        ax.bar(labels,proba[0])
        ax.set_ylabel("Probability")

        st.pyplot(fig)

        st.balloons()


# -----------------------------
# Footer
# -----------------------------
st.markdown("Amina Azam")
st.markdown("🚀 Built with **Streamlit + Scikit-learn**")