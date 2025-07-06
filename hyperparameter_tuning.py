from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_data, train_test_split_data

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name} Best Parameters: {model.best_params_}")
    print(f"{name} MSE: {mse:.2f}")
    print(f"{name} R^2 Score: {r2:.2f}")

def main():
    X_train, X_test, y_train, y_test = train_test_split_data()

    # 1. Ridge Regression
    ridge = Ridge()
    ridge_params = {
        'alpha': [0.1, 1.0, 10.0],
        'solver': ['auto', 'svd', 'cholesky'],
        'fit_intercept': [True, False]
    }
    ridge_grid = GridSearchCV(ridge, ridge_params, cv=5)
    ridge_grid.fit(X_train, y_train)
    evaluate_model("Ridge", ridge_grid, X_test, y_test)

    # 2. Decision Tree
    tree = DecisionTreeRegressor(random_state=42)
    tree_params = {
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
    }
    tree_grid = GridSearchCV(tree, tree_params, cv=5)
    tree_grid.fit(X_train, y_train)
    evaluate_model("Decision Tree", tree_grid, X_test, y_test)

    # 3. Random Forest
    forest = RandomForestRegressor(random_state=42)
    forest_params = {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20]
    }
    forest_grid = GridSearchCV(forest, forest_params, cv=5)
    forest_grid.fit(X_train, y_train)
    evaluate_model("Random Forest", forest_grid, X_test, y_test)

if __name__ == "__main__":
    main()
