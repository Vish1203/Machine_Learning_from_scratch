import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from scipy.stats import spearmanr, kendalltau

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert the dataset to a DataFrame for easier feature selection
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# Calculate the Spearman's rank correlation coefficient between features and the target variable
spearman_corr, _ = spearmanr(df.iloc[:, :-1], df['target'])
spearman_corr = pd.Series(spearman_corr, index=iris.feature_names)

# Calculate the Kendall's rank correlation coefficient between features and the target variable
kendall_corr, _ = kendalltau(df.iloc[:, :-1], df['target'])
kendall_corr = pd.Series(kendall_corr, index=iris.feature_names)

print("Spearman's Rank Correlation Coefficients:")
print(spearman_corr)

print("\nKendall's Rank Correlation Coefficients:")
print(kendall_corr)
