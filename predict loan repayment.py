# Simple project to predict loan repayment using Decision Tree and Random Forest
# Dataset: loan_data.csv
# Goal: Predict whether a loan is fully repaid (not.fully.paid)

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
# Read the loan_data.csv file
data = pd.read_csv('loan_data.csv')

# 2. Prepare the data
# Convert the non-numeric 'purpose' column to numbers (One-Hot Encoding)
data = pd.get_dummies(data, columns=['purpose'], drop_first=True)

# Separate features (X) and target (y)
# Our target is to predict 'not.fully.paid'
X = data.drop('not.fully.paid', axis=1)  # All columns except the target
y = data['not.fully.paid']  # Target column

# 3. Split data into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Decision Tree model
# Create a simple Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)  # Train the model

# Make predictions on test data
dt_predictions = dt_model.predict(X_test)

# Evaluate the Decision Tree model
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("Decision Tree Accuracy:", dt_accuracy)
print("\nDecision Tree Detailed Report:")
print(classification_report(y_test, dt_predictions))

# 5. Random Forest model
# Create a Random Forest model with 100 trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # Train the model

# Make predictions on test data
rf_predictions = rf_model.predict(X_test)

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("\nRandom Forest Accuracy:", rf_accuracy)
print("\nRandom Forest Detailed Report:")
print(classification_report(y_test, rf_predictions))

# 6. Save results
# Save predictions to a file (optional)
results = pd.DataFrame({
    'Actual': y_test,
    'Decision Tree Prediction': dt_predictions,
    'Random Forest Prediction': rf_predictions
})
results.to_csv('predictions.csv', index=False)
print("\nPredictions saved to predictions.csv")