 2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
# Prédiction du cancer du sein
Ce projet utilise des techniques d'apprentissage automatique pour classer les tumeurs mammaires comme malignes ou bénignes, à partir du jeu de données Breast Cancer Wisconsin, en appliquant des classificateurs *Random Forest* et un réglage des hyperparamètres avec *GridSearchCV* pour l’évaluation du modèle.

# Technologies utilisées
 Python 3  
 Pandas  
 Scikit-learn  
 Matplotlib
 
# Objectif
Construire des modèles de classification précis pour aider au dépistage précoce du cancer du sein, en utilisant des méthodes d'apprentissage supervisé.

# Étapes du modèle
 Chargement et prétraitement des données via sklearn.datasets.load_breast_cancer.  
 Division des données en ensembles d'entraînement et de test.  
 Entraînement d’un DecisionTreeClassifier et évaluation de sa précision.  
 Entraînement d’un RandomForestClassifier, optimisation via GridSearchCV, et comparaison des résultats.
 
# Classifieur Random Forest
Paramètres : n_estimators=200, min_samples_leaf=9, random_state=42 
**Précision sur test:** 0.9708  
**Précision sur entraînement:** 0.9698

# Réglage des hyperparamètres
GridSearchCV utilisé avec validation croisée à 5 plis et 72 combinaisons.  
**Meilleurs paramètres :**  
{
  'bootstrap': False,
  'criterion': 'entropy',
  'max_depth': 5,
  'min_samples_leaf': 1,
  'min_samples_split': 8
}


