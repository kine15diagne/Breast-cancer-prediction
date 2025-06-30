# Breast-cancer-prediction
Ce projet est aussi disponible en français : [README.FR.md](README.FR.md)
This project applies machine learning techniques to classify breast tumors as malignant or benign using the Breast Cancer Wisconsin dataset, employing *Random Forest* classifiers and Hyperparameter Tuning
with *GridSearchCV* for evaluation.
# Technologies Used
Python 3
Pandas
Scikit-learn
Matplotlib
# Objective
To build accurate classification models that can assist in early breast cancer detection through supervised learning methods.
# Modeling Steps
Load and preprocess data from sklearn.datasets.load_breast_cancer.
Split data into training and test sets.
Train a DecisionTreeClassifier and evaluate its accuracy.
Train a RandomForestClassifier, optimize it using GridSearchCV, and compare results.
# Random Forest Classifier
n_estimators=200, min_samples_leaf=9, random_state=42
**Test Accuracy:** 0.9708
**Train Accuracy:** 0.9698
# Hyperparameter Tuning
GridSearchCV used with 5-fold cross-validation and 72 combinations.
**Best Parameters:**  
  {
    'bootstrap': False,
    'criterion': 'entropy',
    'max_depth': 5,
    'min_samples_leaf': 1,
    'min_samples_split': 8
  }

