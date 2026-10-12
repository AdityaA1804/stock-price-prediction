"""Model definitions, hyperparameter grids and time-series tuning."""

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

MODEL_GROUPS = {
    "Linear": ["Linear", "Ridge", "Lasso"],
    "Distance / kernel": ["KNN", "SVR"],
    "Trees": ["Decision Tree", "Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost"],
}


def scaled(model):
    """Scaler inside the pipeline, so it is fitted on each training fold only."""
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def model_specs(min_fold_rows, random_state=42, include=None):
    """name -> (estimator, param grid). Pipeline params use the 'model__' prefix.

    KNN neighbour counts larger than the smallest CV training fold are dropped.
    `include` optionally restricts the result to the given model names.
    """
    bagging_grid = {"max_depth": [3, 6, None], "min_samples_leaf": [5, 20], "max_features": [0.3, 1.0]}
    knn_k = [k for k in [5, 10, 20, 30, 50, 75, 100] if k <= min_fold_rows]
    specs = {
        "Linear": (scaled(LinearRegression()), {}),
        "Ridge": (scaled(Ridge()), {"model__alpha": [1, 10, 100, 1000, 10000, 100000]}),
        "Lasso": (scaled(Lasso(max_iter=25000)), {"model__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]}),
        "KNN": (scaled(KNeighborsRegressor()),
                {"model__n_neighbors": knn_k, "model__weights": ["uniform", "distance"]}),
        "SVR": (scaled(SVR(kernel="rbf")),
                {"model__C": [0.0001, 0.001, 0.01, 0.1, 1], "model__epsilon": [0.001, 0.005, 0.01, 0.02],
                 "model__gamma": ["scale", 0.001, 0.0001]}),
        "Decision Tree": (DecisionTreeRegressor(random_state=random_state),
                          {"max_depth": [2, 3, 5], "min_samples_leaf": [10, 30]}),
        "Random Forest": (RandomForestRegressor(n_estimators=300, random_state=random_state), bagging_grid),
        "Extra Trees": (ExtraTreesRegressor(n_estimators=300, random_state=random_state), bagging_grid),
        "Gradient Boosting": (GradientBoostingRegressor(subsample=0.8, random_state=random_state),
                              {"n_estimators": [100, 300], "learning_rate": [0.01, 0.05], "max_depth": [2, 3]}),
        "XGBoost": (XGBRegressor(subsample=0.8, colsample_bytree=0.5, random_state=random_state, n_jobs=1),
                    {"n_estimators": [100, 300], "learning_rate": [0.01, 0.05], "max_depth": [2, 3],
                     "reg_lambda": [1, 10]}),
    }
    if include:
        unknown = set(include) - set(specs)
        if unknown:
            raise ValueError(f"Unknown models: {sorted(unknown)}. Available: {list(specs)}")
        specs = {name: specs[name] for name in specs if name in include}
    return specs


def tune(estimator, grid, x_train, y_train, cv):
    """Grid search on time-series folds; the returned search holds the refitted best model."""
    gs = GridSearchCV(estimator, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    return gs.fit(x_train, y_train)


def format_params(params):
    return ", ".join(f"{k.replace('model__', '')}={v}" for k, v in params.items()) or "-"
