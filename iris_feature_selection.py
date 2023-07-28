import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert the dataset to a DataFrame for easier feature selection
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# Calculate the correlation coefficients between features and the target variable
correlation_coeffs = df.corr()['target'].abs().sort_values(ascending=False)

# Select the top k features with highest absolute correlation coefficients
k = 2
selected_features = correlation_coeffs[1:k+1].index.tolist()

# Split data into train and test sets using the selected features
X_selected = df[selected_features].values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# Train a RandomForestClassifier using the selected features
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions and evaluate the classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
