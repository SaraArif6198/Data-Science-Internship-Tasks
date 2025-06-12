# Data-Science-Internship-Tasks

# 🌸 Task 1 – Exploring and Visualizing the Iris Dataset

## 👨‍💻 Task Objective
This task aims to build essential data science skills, focusing on:
- Loading and understanding structured datasets using **Pandas**
- Visualizing data using **Matplotlib** and **Seaborn**
- Detecting trends, outliers, and relationships in data through **Exploratory Data Analysis (EDA)**
- Building a simple machine learning model to demonstrate understanding of basic model training and evaluation

---

## 🔍 Approach

### 1. Data Loading & Inspection
- Loaded the classic **Iris dataset** using `seaborn.load_dataset("iris")`.
- Used `.shape`, `.columns`, `.info()`, and `.describe()` for structural understanding.

### 2. Data Cleaning & Preparation
- Checked for missing/null values and found none.
- No significant cleaning needed due to the well-structured nature of the dataset.

### 3. Exploratory Data Analysis (EDA)
- Used scatter plots to study relationships between variables (e.g., petal length vs. sepal length).
- Histograms to observe data distribution across all features.
- Box plots to detect outliers and analyze feature spread.

### 4. Model Training & Testing
- Used **Random Forest Classifier** from `scikit-learn`.
- Split the dataset using `train_test_split`.
- Trained the model and predicted on the test set.

### 5. Evaluation Metrics
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-score)

---

## ✅ Results and Insights

### 📈 Data Summary:
- 150 records with 4 numerical features and 1 categorical target (species).

### 🔎 Visual Insights:
- Strong correlation observed between **petal length/width** and **species**.
- Data is fairly balanced across species.
- Some outliers observed in **sepal width**.

### 🧠 Model Performance:
- Achieved **accuracy of ~100%** on test data using Random Forest Classifier.
- All classes (setosa, versicolor, virginica) were correctly identified in the confusion matrix.

---

## 📌 Conclusion
This task provided hands-on experience in data loading, visualization, and model evaluation. The Iris dataset’s simplicity makes it ideal for building foundational skills in data science.  
The strong model accuracy further confirms the dataset's clarity and structure.

---

## 📁 File in This Repository Related To Task 01
- `Task 1-Exploratory Data Analysis in Iris Dataset.ipynb` – Complete Jupyter notebook for this task

---

## 📎 Tools Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook


---


 # 💼 Task 02 – Credit Risk Prediction using Machine Learning

## 👨‍💻 Task Objective
This task aims to build a machine learning pipeline to predict whether a loan application will be approved or not based on applicant information. The project focuses on:
- Cleaning and preprocessing real-world structured data
- Performing Exploratory Data Analysis (EDA)
- Training and evaluating classification models
- Interpreting model outputs and comparing performance

---

## 🔍 Approach

### 1. Data Loading & Inspection
- Loaded the loan dataset using **Pandas**.
- Used `.shape`, `.info()`, `.describe()` to explore structure and get summary statistics.
- Target column: `Loan_Status`.

### 2. Data Cleaning & Preparation
- Handled missing values using mode for categorical columns and median for numerical ones.
- Dropped `Loan_ID` as it has no predictive value.
- Encoded categorical variables using `LabelEncoder`.

### 3. Exploratory Data Analysis (EDA)
- Visualized key features like `Credit_History`, `ApplicantIncome`, `LoanAmount`, and `Education`.
- Identified correlations and trends affecting loan approval.
- Used bar plots, histograms, and box plots with **Seaborn** and **Matplotlib**.

### 4. Model Training
- Split the dataset into training and test sets (80/20 split).
- Trained two models:
  - **Logistic Regression**
  - **Decision Tree Classifier**

### 5. Evaluation Metrics
- Used `accuracy_score`, `confusion_matrix`, and `classification_report`.
- Logistic Regression outperformed Decision Tree in accuracy and F1-score.
- Visualized confusion matrices for better interpretation.

### 6. Predictions
- Predicted and compared loan statuses on test data.
- Displayed actual vs. predicted loan approval status.
- Highlighted correct vs. incorrect predictions visually using color coding.

---

## ✅ Results and Insights

### 📈 Data Summary:
- Dataset included ~600 records with features like income, dependents, education, employment, and loan terms.

### 🔎 Visual Insights:
- Applicants with a **credit history of 1** had significantly higher chances of loan approval.
- Most applicants had low loan amounts and moderate incomes.
- Education and marital status showed minor influence on approval rates.

### 🧠 Model Performance:
- **Logistic Regression Accuracy:** ~79%
- **Decision Tree Accuracy:** ~69%
- Logistic Regression had higher **precision**, **recall**, and **F1-score**, especially for the approved class.

---

## 📌 Conclusion
This project demonstrated a complete machine learning workflow from data preparation to evaluation. Logistic Regression proved more reliable for loan approval prediction in this scenario. The insights can help financial institutions automate and enhance their decision-making processes.

---

## 📁 Files in This Repository
- `Task 02- Credit Risk Prediction.ipynb` – Full Jupyter Notebook with code, EDA, model training, and visualizations

---

## 📎 Tools Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---



# Task 03  🏦 Customer Churn Prediction - Bank Churn Modelling

## 📌 Task Objective
The goal of this project is to build a machine learning model that predicts whether a bank customer is likely to **leave the bank (churn)**. This helps the bank identify at-risk customers and take proactive steps to retain them.

---

## 🧠 Approach

### 1. Dataset
- **Source**: Churn Modelling Dataset (typically includes features like Credit Score, Age, Balance, Geography, etc.)
- **Target**: `Exited` column (0 = Stayed, 1 = Left)

### 2. Steps Followed
- **Data Understanding**: Loaded the dataset and reviewed structure and data types.
- **Data Cleaning & Preparation**:
  - Dropped unnecessary columns like `RowNumber`, `CustomerId`, and `Surname`.
  - Encoded categorical variables:
    - One-Hot Encoding for `Geography`
    - Label Encoding for `Gender`
- **Exploratory Data Analysis (EDA)**:
  - Visualized class distribution, correlations, and feature relationships.
- **Model Building**:
  - Used `RandomForestClassifier` from `scikit-learn`.
  - Trained the model on 80% of the data and tested on 20%.
- **Model Evaluation**:
  - Evaluated performance using accuracy, confusion matrix, and classification report.
- **Feature Importance**:
  - Identified key factors influencing churn (e.g., Age, Balance, IsActiveMember).
- **Customer Prediction**:
  - Displayed the customers likely to churn using model predictions.

---

## 📊 Results and Insights

- **Model Accuracy**: ~87%
- **Precision & Recall**:
  - High precision for both churned and non-churned classes.
  - Slightly lower recall for customers who actually churned.
- **Important Features**:
  - Age and Account Balance strongly influence churn behavior.
  - Inactive members are significantly more likely to leave.
- **Business Insight**:
  - The bank can use this model to identify high-risk customers and launch targeted campaigns to reduce churn and improve customer retention.

---

## 📂 Files Included
- `Task 03-Customer_Churn_Prediction.ipynb` – Full Jupyter Notebook with step-by-step analysis and model

---

## ✅ Tools & Libraries Used
- Python
- Pandas
- Matplotlib & Seaborn
- Scikit-learn

---


# Task 04-🩺 Medical Insurance Charges Prediction

## 🎯 Task Objective

The objective of this project is to build a predictive model using **Linear Regression** that estimates **medical insurance charges** based on personal attributes such as:

- Age  
- Sex  
- BMI (Body Mass Index)  
- Number of Children  
- Smoking Status  
- Region  

The dataset used is the **Medical Cost Personal Dataset**, which is commonly available on [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance).

---

## 🛠️ Approach

The project was completed in several systematic steps using Python and Jupyter Notebook:

### 📌 Step-by-Step Workflow:

1. **Introduction and Problem Statement**  
   - Defined the goal of predicting medical insurance charges and understanding influencing factors.

2. **Dataset Understanding**  
   - Loaded and explored the dataset structure using `pandas`.

3. **Data Cleaning and Preparation**  
   - Verified missing values and data types.  
   - Encoded categorical variables (`sex`, `smoker`, `region`) using one-hot encoding.

4. **Exploratory Data Analysis (EDA)**  
   - Used `matplotlib` and `seaborn` for visualization.  
   - Plotted:
     - Correlation heatmap  
     - BMI vs Charges  
     - Age vs Charges  
     - Charges by Smoking Status  

5. **Model Training and Testing**  
   - Split the dataset into training and testing sets (80/20 split).  
   - Trained a **Linear Regression** model using `scikit-learn`.

6. **Model Evaluation**  
   - Calculated:
     - **Mean Absolute Error (MAE)**: `4181.19`  
     - **Root Mean Squared Error (RMSE)**: `5796.28`  

7. **Classifying High vs Low Charges**  
   - Predicted all charges and labeled them as **High** or **Low** based on median value.  
   - Visualized distribution of high/low charges using boxplots and count plots.

---

## 📈 Results and Insights

- **Smoking** is the strongest contributor to high insurance charges.
- **Age** and **BMI** show a positive correlation with charges.
- The model reasonably predicts charges with MAE ≈ 4181 and RMSE ≈ 5796.
- People who **smoke and have higher BMI** are much more likely to have **high insurance costs**.
- Classification and visualization helped in identifying which people fall into high or low risk categories.

---

## ✅ Conclusion

This project illustrates a full **machine learning workflow**:
- From data cleaning to EDA
- Building and evaluating a model
- Drawing real-world insights from predictive analytics

The project is a good starting point for developing more advanced models with additional features like medical history, genetic data, or location-based trends.

---

## 📂 Files Included
- `Task 04-Predicting Insurance Claim Amount.ipynb` – Full Jupyter Notebook with step-by-step analysis and model

---

## 📚 Libraries Used

- `pandas`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`  
- `numpy`

---


# Task 05-📊 Personal Loan Acceptance Prediction

## 🎯 Task Objective

The goal of this project is to develop a predictive model that identifies which customers are likely to subscribe to a **term deposit** based on their demographic, financial, and contact information. This helps financial institutions optimize marketing strategies and improve loan acceptance rates.

The dataset includes features such as:

- Age  
- Job Type  
- Marital Status  
- Education  
- Default Status  
- Account Balance  
- Housing Loan  
- Personal Loan  
- Contact Communication  
- Campaign Details  
- Previous Contact Outcome  
- And more  

---

## 🛠️ Approach

The project followed a structured machine learning workflow using Python and Jupyter Notebook:

### 📌 Step-by-Step Workflow:

1. **Data Understanding & Preprocessing**  
   - Explored dataset features and cleaned the data.  
   - Encoded categorical variables for model compatibility.  
   - Checked and handled class imbalance in the target variable.

2. **Exploratory Data Analysis (EDA)**  
   - Visualized age distribution, job types, marital status, and loan acceptance patterns.  
   - Identified that older customers and certain job types are more likely to accept loans.

3. **Model Training and Testing**  
   - Split the data into training and testing sets (80/20).  
   - Built and trained two models:  
     - Logistic Regression  
     - Decision Tree Classifier

4. **Model Evaluation**  
   - Evaluated models using accuracy, confusion matrix, precision, recall, and F1-score.  
   - Logistic Regression achieved ~81% accuracy; Decision Tree reached ~79%.

5. **Prediction & Interpretation**  
   - Used the Decision Tree model to predict subscription status for new customers.  
   - Converted predictions to clear "Yes"/"No" labels for easy understanding.

---

## 📈 Results and Insights

- Most customers are aged between 20 and 60, with loan acceptance skewed slightly towards older age groups.  
- Jobs such as blue-collar, management, and technician dominate the dataset.  
- The dataset shows imbalance with fewer customers subscribing to term deposits.  
- Both Logistic Regression and Decision Tree models delivered reliable and balanced predictions.  
- Decision Tree model predictions help clearly identify which customers are likely to subscribe, assisting targeted marketing.

---

## ✅ Conclusion

This project demonstrates how customer data can be effectively leveraged with machine learning to predict term deposit subscriptions. The insights gained support financial institutions in:

- Targeting marketing campaigns more efficiently  
- Improving the acceptance rate of loan offers  
- Enhancing customer engagement through predictive analytics  

This workflow provides a foundation for further enhancements with additional data and advanced algorithms.

---

## 📂 Files Included
- `Task 05-Personal Loan Acceptance Prediction.ipynb` – Full Jupyter Notebook with step-by-step analysis and model

---

## 📚 Libraries Used

- `pandas`  
- `numpy`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`

---

## 📁 Dataset Source

- Task01-  https://archive.ics.uci.edu/dataset/53/iris
- Task02-  https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
- Task03-  https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
- Task04-  https://www.kaggle.com/datasets/mirichoi0218/insurance
- Task05-  https://archive.ics.uci.edu/ml/datasets/Bank+Marketing


## 🚀 Author
- **[Sara Arif]**
- Data Science Intern


# My License Repository

This repository contains content protected under the Creative Commons BY-NC-ND 4.0 License.

## 📜 License

This project is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International** license.

- 📖 [Read Full License](https://creativecommons.org/licenses/by-nc-nd/4.0/)
- ✅ You can view and share.
- ❌ No commercial use.
- ❌ No modifications allowed.

