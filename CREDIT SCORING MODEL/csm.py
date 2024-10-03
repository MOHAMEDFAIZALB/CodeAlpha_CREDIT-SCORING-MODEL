# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

# Load the dataset
# Assuming you have a CSV file with features like 'income', 'debt', 'credit_history', and target 'creditworthy'
data = pd.read_csv('credit_data.csv')

# Display first few rows of the dataset
print(data.head())

# Preprocessing: Handle missing values
# Impute missing numerical values with mean
imputer = SimpleImputer(strategy='mean')
data[['income', 'debt', 'credit_history']] = imputer.fit_transform(data[['income', 'debt', 'credit_history']])

# Feature selection
X = data[['income', 'debt', 'credit_history']]  # Features
y = data['creditworthy']  # Target (creditworthiness: 1 for good credit, 0 for bad credit)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for algorithms like Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_log_reg = log_reg.predict(X_test_scaled)

# Evaluate Logistic Regression model
print("Logistic Regression Model")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg)}")
print(classification_report(y_test, y_pred_log_reg))

# Model 2: Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)  # No need to scale for Random Forest

# Predict on the test set
y_pred_rf = rf.predict(X_test)

# Evaluate Random Forest model
print("Random Forest Classifier Model")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(classification_report(y_test, y_pred_rf))
