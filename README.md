# Diabetes-Prediction-using-Machine-Learning-with-Python

=> Project Overview
- This project uses Machine Learning to predict whether a patient is diabetic or non-diabetic based on various health metrics. The prediction is made using a Support Vector Machine (SVM) classifier, and the - project is implemented in Python using popular libraries like Pandas, NumPy, and Scikit-learn. The dataset used is the Pima Indians Diabetes Database.

=> Objective
- The goal is to build a machine learning model that can predict the onset of diabetes in patients based on their health indicators, aiding in early detection and management of the condition.

=> Dataset Information
- The dataset consists of medical records for females of Pima Indian heritage, including the following health metrics:
    - Pregnancies: Number of pregnancies the patient has had.
    - Glucose: Plasma glucose concentration (mg/dL).
    - Blood Pressure: Diastolic blood pressure (mm Hg).
    - Skin Thickness: Triceps skinfold thickness (mm).
    - Insulin: 2-hour serum insulin (mu U/ml).
    - BMI: Body mass index (weight in kg/(height in m)^2).
    - Diabetes Pedigree Function: A function representing the likelihood of diabetes based on family history.
    - Age: The patient’s age (years).
    - Outcome: Target variable indicating whether the patient is diabetic (1) or non-diabetic (0).

=> Dataset Summary
- Total Records: 768
- Features: 8
- Target: 1 (Outcome)
- Classes: 0 (Non-diabetic), 1 (Diabetic)

=> Project Workflow

1. Data Collection & Exploration
- The dataset is loaded and examined to understand its structure, size, and statistical properties. The data includes important health metrics, and a preliminary analysis is performed to check for class balance (number of diabetic vs. non-diabetic patients) and any missing values.

2. Data Preprocessing
- The data is prepared for machine learning by scaling the feature values. Standardization is applied to bring all features to a similar scale, ensuring that the machine learning model isn't biased towards features with larger numerical ranges.

3. Model Selection & Training
- A Support Vector Machine (SVM) classifier with a linear kernel is chosen for its effectiveness in binary classification tasks. The dataset is split into training and testing sets to evaluate the model’s performance. The training data is used to fit the model, while the test data helps assess its generalization ability.

4. Model Evaluation
- After training, the model is evaluated based on its accuracy on both the training and testing sets. Accuracy scores are calculated to determine how well the model performs in predicting diabetes.

5. Prediction System
- A prediction system is implemented to allow users to input new data and get real-time predictions about whether a patient is diabetic or not. This prediction system applies the same data preprocessing steps before feeding the data into the trained model.

=> Results
- The model achieved a reasonable accuracy on both training and testing datasets, showing its potential in predicting diabetes effectively. The accuracy scores indicate that the model generalizes well to unseen data, making it a useful tool for diabetes prediction.

=> Conclusion
- This project demonstrates the process of building a machine learning model to predict diabetes based on health metrics. By employing Support Vector Machines and data preprocessing techniques, the model can assist in early detection of diabetes, providing valuable support for healthcare decisions.
