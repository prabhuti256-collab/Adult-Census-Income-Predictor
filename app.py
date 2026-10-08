import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Adult Census Income Predictor",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("models/income_model.pkl")


# ============================================================
# TITLE
# ============================================================

st.title("💰 Adult Census Income Predictor")

st.write(
    "Predict whether a person's annual income is likely to be "
    "**≤50K or >50K** based on census information."
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Enter Person Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=17,
        max_value=90,
        value=30
    )

    workclass = st.selectbox(
        "Workclass",
        [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked"
        ]
    )

    education = st.selectbox(
        "Education",
        [
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "9th",
            "7th-8th",
            "12th",
            "Masters",
            "1st-4th",
            "10th",
            "Doctorate",
            "5th-6th",
            "Preschool"
        ]
    )

    education_num = st.number_input(
        "Education Number",
        min_value=1,
        max_value=16,
        value=10
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse"
        ]
    )


with col2:

    occupation = st.selectbox(
        "Occupation",
        [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces"
        ]
    )

    relationship = st.selectbox(
        "Relationship",
        [
            "Wife",
            "Own-child",
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Unmarried"
        ]
    )

    race = st.selectbox(
        "Race",
        [
            "White",
            "Black",
            "Asian-Pac-Islander",
            "Amer-Indian-Eskimo",
            "Other"
        ]
    )

    sex = st.selectbox(
        "Sex",
        [
            "Male",
            "Female"
        ]
    )

    native_country = st.selectbox(
        "Native Country",
        [
            "United-States",
            "Mexico",
            "Philippines",
            "Germany",
            "Canada",
            "Puerto-Rico",
            "El-Salvador",
            "India",
            "China",
            "Cuba",
            "England",
            "Jamaica",
            "Other"
        ]
    )


with col3:

    fnlwgt = st.number_input(
        "Final Weight (fnlwgt)",
        min_value=10000,
        max_value=1500000,
        value=180000
    )

    capital_gain = st.number_input(
        "Capital Gain",
        min_value=0,
        max_value=100000,
        value=0
    )

    capital_loss = st.number_input(
        "Capital Loss",
        min_value=0,
        max_value=5000,
        value=0
    )

    hours_per_week = st.number_input(
        "Hours per Week",
        min_value=1,
        max_value=99,
        value=40
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔮 Predict Income",
    use_container_width=True
):

    input_data = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education-num": education_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success("🎉 Predicted Income: **>50K**")

    else:

        st.info("📊 Predicted Income: **≤50K**")


    # Probability display

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Probability of ≤50K",
            f"{probability[0] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Probability of >50K",
            f"{probability[1] * 100:.2f}%"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Adult Census Income Predictor | Machine Learning Project"
)

