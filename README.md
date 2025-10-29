# Loan-Status-Prediction-using-Machine-learning-in-Python
Problem Statement

Financial institutions receive thousands of loan applications daily. Many applicants are rejected due to poor credit history, low income, or high debt.
Manually evaluating each loan request is time-consuming and error-prone.

Goal: Build a Machine Learning–powered application to automatically predict whether a loan application should be Approved or Rejected based on applicant data such as income, loan amount, age, credit score, and employment history.

Solution Overview

This project implements an intelligent loan approval system built using Python (Flask) and Machine Learning.
The system validates applicant inputs, performs feature-based analysis, and predicts loan approval probability using trained ML models.
Users can visualize model accuracy, F1-score, and predictions through an interactive web interface.
<img width="1913" height="992" alt="image" src="https://github.com/user-attachments/assets/b979fccb-0df8-4e45-9c4d-cbe14f24f87e" />

<img width="1916" height="992" alt="image" src="https://github.com/user-attachments/assets/ecd0dc2d-816c-4f83-87fc-384fdc3c2b3c" />

Key Features

Data Validation:
Ensures all inputs meet eligibility rules (e.g., age > 18, income ≥ ₹35,000, credit score 100–850).

Multi-Model Support:
Compare multiple algorithms — Logistic Regression, Decision Tree, Random Forest, KNN, and Naive Bayes — to analyze performance metrics.

Interactive Interface:
Built with Flask web forms (previously Swing-based GUI) for a clean, user-friendly experience.

Performance Metrics:
Displays Accuracy, Precision, Recall, and F1-Score for each algorithm to help evaluate model.

Real-Time Prediction:
Predicts instantly whether a loan will be approved or not based on user inputs.

Tech Stack

Programming Language - Python

ML Libraries	- Scikit-learn, Pandas, NumPy

Frontend	- HTML, CSS (via Flask Templates)

Backend	- Flask (Python Web Framework)

Data Handling	- Pandas, CSV

Visualization	- Matplotlib

IDE - 	VS Code 


Application Flow

Step 1: Input Validation

User enters loan-related details such as age, income, loan amount, credit score, and years of employment.
If invalid data is entered, an alert guides the user to correct it (as shown in screenshots).

Step 2: Model Selection

Choose from different ML algorithms and instantly view model metrics.

Step 3: Loan Status Prediction

Displays whether the loan is likely to be approved or rejected along with performance statistics.
