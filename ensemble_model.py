from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Data Preparation
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Model Selection
models = {
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'LinearRegression': LinearRegression()
}

best_params = {}  # Dictionary to store the best parameters for each model

for name, model in models.items():
    param_grid = {}
    if name == 'RandomForest':
        param_grid['n_estimators'] = [50, 100, 150]
        param_grid['max_depth'] = [None, 5, 10]
    elif name == 'GradientBoosting':
        param_grid['n_estimators'] = [50, 100, 150]
        param_grid['max_depth'] = [3, 5, 7]
    elif name == 'LinearRegression':
        param_grid = {'fit_intercept': [True, False]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params[name] = grid_search.best_params_  # Store the best parameters
    print(f"Best parameters for {name}: {best_params[name]}")
    print(f"Best MSE for {name}: {-grid_search.best_score_}")

# Step 3: Ensemble Construction using best parameters
base_models = [
    ('RandomForest', RandomForestRegressor(**best_params['RandomForest'])),
    ('GradientBoosting', GradientBoostingRegressor(**best_params['GradientBoosting'])),
    ('LinearRegression', LinearRegression(**best_params['LinearRegression']))
]
ensemble = VotingRegressor(estimators=base_models)

# Step 4: Evaluation
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
ensemble_mse = mean_squared_error(y_test, y_pred)
print(f"Ensemble MSE: {ensemble_mse}")
