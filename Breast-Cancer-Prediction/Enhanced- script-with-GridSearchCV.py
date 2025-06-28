# =======================================================================
# Enhanced script with GridSearchCV to optimize a RandomForestClassifier
# =======================================================================
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # To save the best model if desired

# 1. Load data
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 2. Define the base classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 3. Define the hyperparameter grid
param_grid = {
    "max_depth": [None, 3, 5],
    "min_samples_split": [4, 8],
    "min_samples_leaf": [1, 5],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy", "log_loss"],
}

# 4. Configure GridSearchCV
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                     # 5-fold cross-validation
    scoring="accuracy",
    n_jobs=-1,                # Use all available cores
    verbose=1,
)

# 5. Fit the grid
grid.fit(X_train, y_train)

print("\nBest hyperparameters found:")
print(grid.best_params_)
print(f"CV Score = {grid.best_score_:.4f}")

# 6. Evaluate on the test set using the best model
best_rf = grid.best_estimator_
y_pred = best_rf.predict(X_test)
acc_test = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy = {acc_test:.4f}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# 7. (Optional) Save the model for production
joblib.dump(best_rf, "best_random_forest.pkl")
