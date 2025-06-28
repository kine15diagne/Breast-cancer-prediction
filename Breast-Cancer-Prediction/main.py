
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Breast Cancer Wisconsin dataset
cancer_data = load_breast_cancer()
data = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
data['target'] = cancer_data.target

# Split the data into training and testing sets
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Random Forest model with 200 trees
model = RandomForestClassifier(n_estimators=200, random_state=42, min_samples_leaf=9)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = model.predict(X_train)

# Calculate the model's accuracy on the training set
accuracy_train = accuracy_score(y_train, y_train_pred)
print(f'Random Forest model accuracy on Train: {accuracy_train}')

# Make predictions on the test set
y_test_pred = model.predict(X_test)

# Calculate the model's accuracy on the test set
accuracy_test = accuracy_score(y_test, y_test_pred)
print(f'Random Forest model accuracy on Test: {accuracy_test}')
