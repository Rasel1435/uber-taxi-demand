import mlflow
import config
import mlflow.sklearn

from typing import Union
from numpy import ndarray
from zenml import step, client
from pandas import DataFrame, Series
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from logs.logs import configure_logger
logger = configure_logger()


# Experiment Tracker
tracker = client.Client().active_stack.experiment_tracker


@step(enable_cache=True, experiment_tracker=tracker.name)
def evaluate(X: Union[DataFrame, ndarray], y: Union[Series, ndarray], label='TEST') -> bool:
    """
    This step evaluate the model.

    Args:
        data (Union[pd.DataFrame, None]): The input data.

    Returns:
        bool: True if the The model is evaluate successfully, False otherwise.
    """
    try:
        logger.info(f'==> Processing evaluate()')

        mlflow.set_experiment("Uber Taxi Demand")
        with mlflow.start_run():
            model = mlflow.sklearn.load_model(f'{config.MODEL_NAME} -> XGBoost')
            y_pred = model.predict(X)
            
            # Calculate and log the evaluation metric
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            mape = mean_absolute_percentage_error(y, y_pred) * 100
            r2 = r2_score(y, y_pred) * 100
            
            #Log Matrics
            mlflow.log_metrics({
                f'MSE{label}': mse,
                f'MAE{label}': mae,
                f'MAPE{label}': mape,
                f'R2_SCORE{label}': r2
            })
        logger.info(f'==> Successfully processed evaluate()')
        return True
    except Exception as e:
        logger.error(f'in evaluate(): {e}')
        return False