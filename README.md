# 💰 Adult Census Income Predictor

An end-to-end **Machine Learning project** that predicts whether an individual's annual income is **<=50K or >50K** using demographic, educational, employment, and financial features from the Adult Census dataset.

The project includes data preprocessing, exploratory data analysis, visualization, multiple classification models, model evaluation, and an interactive **Streamlit web application**.

## 🚀 Project Overview

The Adult Census Income Predictor uses machine learning classification algorithms to predict income categories based on information such as:

* Age
* Workclass
* Education
* Education Number
* Marital Status
* Occupation
* Relationship
* Race
* Sex
* Capital Gain
* Capital Loss
* Hours per Week
* Native Country
* Final Weight

## 🧠 Machine Learning Models

The following classification algorithms were trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

The best-performing model based on **F1-score** is selected and saved for making predictions.

## 📊 Dataset

The project uses the **Adult Census Income dataset**.

After data cleaning:

* **Records:** 30,162
* **Input Features:** 14
* **Target Variable:** Income
* **Classes:**

  * `<=50K`
  * `>50K`

### Class Distribution

* `<=50K`: 75.11%
* `>50K`: 24.89%

## 🔧 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit

## 📁 Project Structure

```text
Adult-Census-Income-Predictor/
│
├── dataset/
│   ├── adult.csv
│   ├── adult.data
│   ├── adult.test
│   ├── adult.names
│   └── ...
│
├── models/
│   └── income_model.pkl
│
├── visualizations/
│   ├── age_distribution.png
│   ├── education_vs_income.png
│   ├── hours_vs_income.png
│   ├── income_by_sex.png
│   └── income_distribution.png
│
├── app.py
├── create_dataset.py
├── explore_data.py
├── predict.py
├── train_model.py
├── visualize_data.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔄 Machine Learning Workflow

```text
Adult Census Dataset
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Data Visualization
        ↓
Feature Preprocessing
        ↓
Train/Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Best Model Selection
```
