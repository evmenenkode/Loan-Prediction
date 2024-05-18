# Loan Status Predictor

## Overview
This repository contains code for predicting loan status based on various features using machine learning algorithms. The primary goal is to automate the loan approval process by accurately predicting whether a loan application should be approved or rejected.

## Data Exploration
The initial step involves exploring the dataset to understand its structure and characteristics. We analyze features such as applicant income, co-applicant income, loan amount, loan amount term, credit history, etc. Missing values are handled by dropping rows with NaN values. Categorical variables are converted into numerical values for further analysis.

## Model Selection
We employ various machine learning algorithms such as KNeighborsClassifier, LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, XGBClassifier, CatBoostClassifier, and LGBMClassifier to build predictive models. Grid search is utilized to find the best parameters for each model, and the models are evaluated using cross-validation.

## Model Evaluation
The performance of each model is evaluated based on accuracy, precision, recall, and F1-score metrics using cross-validation. The RandomForestClassifier model demonstrates the highest overall accuracy among all models.

## Training and Testing
The RandomForestClassifier model is chosen for training and testing on the dataset. The dataset is split into training and testing sets, and the model is trained on the training set. The trained model is then tested on the testing set to evaluate its performance.

## Conclusion
The RandomForestClassifier model demonstrates good performance with high precision, recall, and F1-score. It is chosen as the preferred model for predicting loan status due to its ability to generalize well to new data and its resistance to overfitting.

## Usage
A function loan_status_predictor(input_array) is provided, which takes an input array of features and returns the predicted loan status ('Loan Approved' or 'Loan Rejected').

## Files
- loan.csv: Dataset containing loan application information.
- loan_status_predictor.py: Python script containing code for data analysis, model building, evaluation, and prediction.
- README.md: Markdown file providing an overview of the project and instructions for usage.
## Dependencies
- Python 3.x
- Libraries: numpy, pandas, seaborn, matplotlib, scikit-learn, imbalanced-learn, lightgbm, xgboost, catboost
## How to Use
To use the loan status predictor function, follow these steps:

1. Ensure Python and the required libraries are installed.
2. Clone or download this repository to your local machine.
3. Run the loan_status_predictor.py script.
4. Use the loan_status_predictor(input_array) function with appropriate input features to predict loan status.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any suggestions or improvements
