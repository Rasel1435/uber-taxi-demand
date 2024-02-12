import pandas as pd
import mlflow.sklearn
import mlflow
import warnings
import xgboost as xgb
import logging

from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error, r2_score

warnings.filterwarnings('ignore')

from logs.logs import configure_logger
logger = configure_logger()

#Loading data
def getData(path: str) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        logger.error(f"in getData(): {e}")

def splitting() -> tuple:
    global df
    try:
        X_train = df.drop(columns=["taxi_demand",])
        y_train = df.taxi_demand
        df2 = pd.read_parquet(r'../data/feature-2023.parquet')
        X_test = df2.drop(columns=["taxi_demand",])
        y_test = df2.taxi_demand
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f'in splitting(): {e}')

def myModelxgb(X_train, X_test, y_train, y_test) -> xgb.XGBRegressor:
    try:
        mlflow.set_experiment("TimeSeries")
        with mlflow.start_run():
            x_model = xgb.XGBRegressor()
            param_dist = {
                'max_depth': randint(1, 16),
                'n_estimators': randint(100, 600),
                'min_child_weight': randint(1, 16),
                'gamma': [0, 0.1, 0.2],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'nthread': randint(1, 16),
            }
            # run a randomized search
            n_iter_search = 20
            random_search = RandomizedSearchCV(x_model, param_distributions=param_dist,
                                               n_iter=n_iter_search, random_state=42)
            # fit the model
            random_search.fit(X_train, y_train)
            # Predict on the test set using the best estimator from the grid search 2023
            y_pred = random_search.best_estimator_.predict(X_train)
            
            # Log parameters 
            # mlflow.log_params(random_search.best_params_)
            
            # Calculate and log the evaluation metric (e.g., RMSE) 2022
            rmse = mean_squared_error(y_train, y_pred, squared=False)
            mape = mean_absolute_percentage_error(y_train, y_pred)
            mae = mean_absolute_error(y_train, y_pred)
            r2 = r2_score(y_train, y_pred)

            #Log Matrics 2022
            mlflow.log_metrics({
                "RMSE_train": rmse,
                "MAE_train": mae,
                "MAPE0_train": mape,
                "R2_SCORE_train": r2
            })
            
            # Predict on the test set using the best estimator from the grid search 2023
            y_pred = random_search.best_estimator_.predict(X_test)
            
            # Log parameters 
            mlflow.log_params(random_search.best_params_)
            
            # Calculate and log the evaluation metric (e.g., RMSE) 2023
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            #Log Matrics
            mlflow.log_metrics({
                "RMSE": rmse,
                "MAE": mae,
                "MAPE0": mape,
                "R2_SCORE": r2
            })


            # Saving the best model obtained after hyperparameter tuning
            mlflow.sklearn.log_model(random_search.best_estimator_, 'XGBoost_best_model')

            return random_search.best_estimator_
    except Exception as e:
        logger.error(f"in myModelxgb(): {e}")

if __name__ == '__main__':
    df = getData(r'../data/feature-2022.parquet')
    # df2 = getData(r'../data/feature-2023.parquet')
    X_train, X_test, y_train, y_test = splitting()
    best_model = myModelxgb(X_train, X_test, y_train, y_test)
    # we have to use 'best_model' for further predictions or inference
