import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Univariate feature selection using SelectKBest
# Select the top 2 features based on the chi-square test
selector = SelectKBest(score_func=chi2, k=2)
X_train_kbest = selector.fit_transform(X_train, y_train)
X_test_kbest = selector.transform(X_test)

# 2. Recursive Feature Elimination (RFE)
# Select the top 2 features using RFE with a RandomForestClassifier as the estimator
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
rfe_selector = RFE(estimator, n_features_to_select=2)
X_train_rfe = rfe_selector.fit_transform(X_train, y_train)
X_test_rfe = rfe_selector.transform(X_test)

# 3. Feature Importance from Tree-based models
# Use RandomForestClassifier to rank feature importance
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
sorted_indices = np.argsort(importances)[::-1]  # Sort features in descending order of importance
top_feature_indices = sorted_indices[:2]
X_train_imp = X_train[:, top_feature_indices]
X_test_imp = X_test[:, top_feature_indices]

# Helper function to train and evaluate a classifier
def evaluate_classifier(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

# Create and evaluate classifiers with different feature selection techniques
classifiers = {
    'All Features': RandomForestClassifier(n_estimators=100, random_state=42),
    'SelectKBest': RandomForestClassifier(n_estimators=100, random_state=42),
    'RFE': RandomForestClassifier(n_estimators=100, random_state=42),
    'Feature Importance': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Using all features
acc_all = evaluate_classifier(classifiers['All Features'], X_train, X_test, y_train, y_test)

# Using SelectKBest features
acc_kbest = evaluate_classifier(classifiers['SelectKBest'], X_train_kbest, X_test_kbest, y_train, y_test)

# Using RFE-selected features
acc_rfe = evaluate_classifier(classifiers['RFE'], X_train_rfe, X_test_rfe, y_train, y_test)

# Using feature importance-selected features
acc_imp = evaluate_classifier(classifiers['Feature Importance'], X_train_imp, X_test_imp, y_train, y_test)

# Display the results
results = pd.DataFrame({'Accuracy': [acc_all, acc_kbest, acc_rfe, acc_imp]},
                       index=['All Features', 'SelectKBest', 'RFE', 'Feature Importance'])
print(results)
