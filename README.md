# Loan Repayment Prediction with Decision Tree and Random Forest

This is a simple machine learning project to demonstrate how to use Decision Tree and Random Forest algorithms to predict whether a loan will be fully repaid.

## Dataset
The loan_data.csv dataset contains information about loans, including:
- credit.policy: Whether the customer meets credit criteria (1 = yes, 0 = no).
- fico: The customer's credit score.
- not.fully.paid: Our target (0 = loan fully repaid, 1 = not fully repaid).

## Goal
The goal is to predict whether a loan will be fully repaid using two algorithms:
- Decision Tree: A simple model that splits data into branches.
- Random Forest: A more advanced model that combines multiple decision trees.

## Files
- loan_data.csv: The dataset.
- loan_prediction.py: Python code for training and evaluating models.
- predictions.csv: Prediction results (generated after running the code).

## How to Run
1. Ensure Python and the required libraries are installed: pip install pandas scikit-learn
2. Place loan_data.csv in the project folder.
3. Run the code: python loan_prediction.py
4. The output (model accuracy and detailed reports) will be printed, and predictions will be saved to predictions.csv.

## Results
- Decision Tree Accuracy: Around 70-80% (depends on the data).
- Random Forest Accuracy: Usually better, around 80-85%, since it uses multiple trees.

## Why This Project?
This project is designed for beginners to show how to solve a real-world problem (loan repayment prediction) with simple algorithms. What I learned:
- Data preparation (e.g., handling non-numeric columns).
- Training and evaluating machine learning models.
- Comparing two different algorithms.

## Next Steps
- Improve models by tuning parameters (e.g., tree depth or number of trees).
- Add more analysis, like feature importance.
- Try larger or more complex datasets.

---

Built by: [Maryam Abdi]
