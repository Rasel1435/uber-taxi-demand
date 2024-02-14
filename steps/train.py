import mlflow.sklearn
import mlflow
import config

from zenml import step, client
from typing import Union, Dict
from scipy.stats import randint
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from logs import configure_logger
logger = configure_logger()

# Experiment Tracker
tracker = client.Client().active_stack.experiment_tracker
@step(name='Train XGBoostRegressor', experiment_tracker=tracker.name)
def trainXGB(data: Dict) -> Union[BaseEstimator, None]:
    """
    This step trains a model using the xgboost library

    Args:
        data (Union[pd.DataFrame, None]): The input data

    Returns:
        XGBoost: The trained model
    """
    try:
        logger.info(f'==> Processing trainXGB()')
        X_train = data['X_train']
        y_train = data['y_train']

        xgb_model = XGBRegressor()
        param_dist = {
            'max_depth': randint(1, 16),
            'n_estimators': randint(100, 600),
            'min_child_weight': randint(1, 16),
            'gamma': [0, 0.1, 0.2],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'nthread': randint(1, 16),
        }
        # run a randomized search
        n_iter_search = 25
        random_search = RandomizedSearchCV(
            xgb_model, param_distributions=param_dist,
            n_iter=n_iter_search, random_state=42, verbose=1
        )
        # fit the model
        random_search.fit(X_train, y_train)
        # Log parameters
        mlflow.log_params(random_search.best_params_)
        # Saving the best model obtained after hyperparameter tuning
        mlflow.sklearn.log_model(
                random_search.best_estimator_, f'{config.MODEL_NAME}-XGBoost')

        logger.info(f'==> Successfully processed trainXGB()')
        return random_search.best_estimator_
    except Exception as e:
        logger.error(f'in trainXGB(): {e}')
        return None
# zenml artifact-store register my_s3_store --flavor=s3     --path=s3://my_bucket --client_kwargs='{"endpoint_url": "http://my-s3-endpoint"}'