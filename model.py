import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame from the Iris dataset
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                       columns= iris['feature_names'] + ['target'])

# Split dataset into features and target variable
X = iris_df.drop('target', axis=1) # Features
y = iris_df['target'] # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, cv=5)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

# Get the best estimator
best_clf = grid_search.best_estimator_

# Predict the response for test dataset using the best estimator
y_pred_tuned = best_clf.predict(X_test)

# Model Accuracy after tuning
print("\nAccuracy after tuning:", accuracy_score(y_test, y_pred_tuned))

# Classification Report after tuning
print("\nClassification Report after tuning:")
print(classification_report(y_test, y_pred_tuned))

# Confusion Matrix after tuning
print("\nConfusion Matrix after tuning:")
print(confusion_matrix(y_test, y_pred_tuned))

# Plot the decision tree of the best estimator
plt.figure(figsize=(12,8))
plot_tree(best_clf, filled=True, feature_names=iris['feature_names'], class_names=iris['target_names'])
plt.show()
