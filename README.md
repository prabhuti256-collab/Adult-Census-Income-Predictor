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
        ↓
Model Saving
        ↓
Streamlit Application
        ↓
Income Prediction
```

## 📈 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

The model with the best **F1-score** is selected as the final prediction model.

## 🌐 Streamlit Application

The project includes an interactive Streamlit interface where users can enter an individual's:

* Age
* Workclass
* Education
* Occupation
* Marital Status
* Relationship
* Race
* Sex
* Capital Gain/Loss
* Working Hours
* Native Country
* Other census information

The application then predicts whether the estimated annual income is:

**💵 <=50K**

or

**💰 >50K**

It also displays the model's prediction probabilities.

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/prabhuti256-collab/Adult-Census-Income-Predictor.git
```

Navigate to the project:

```bash
cd Adult-Census-Income-Predictor
```

Create a virtual environment:

```bash
py -3.12 -m venv venv
```

Activate the environment on Windows:

```powershell
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 🧪 Run Prediction from Python

You can also run the prediction script:

```bash
python predict.py
```

## 📊 Data Analysis

The project includes visualizations for:

* Income distribution
* Age distribution
* Education vs income
* Working hours vs income
* Income distribution by sex

These visualizations help identify patterns and relationships within the dataset.

## 🔮 Future Improvements

* Hyperparameter tuning
* Cross-validation
* Feature importance analysis
* Interactive model comparison
* Improved Streamlit dashboard
* SHAP-based model explainability
* Cloud deployment
* Real-time prediction API

## 👩‍💻 Author

**Prabhuti**

B.Tech AI & ML Student

GitHub: https://github.com/prabhuti256-collab

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.
