import mlflow.sklearn
import mlflow

from typing import Dict
from zenml import step, client
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from logs import configure_logger
logger = configure_logger()

tracker = client.Client().active_stack.experiment_tracker
# Step to evaluate the model
@step(name='evaluate', experiment_tracker=tracker.name)
def evaluate(data: Dict, model: BaseEstimator, label='TEST') -> bool:
    """
    This step evaluates the model.

    Args:
        data (Union[pd.DataFrame, None]): The input data.

    Returns:
        bool: True if the model is evaluated successfully, False otherwise.
    """
    try:
        logger.info(f'==> Processing evaluate() on {label}')
        # Split the data into training and testing sets
        split = label.lower()
        X = data[f'X_{split}']
        y = data[f'y_{split}']
        y_pred = model.predict(X)
        # Calculate and log the evaluation metric
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred) * 100
        r2 = r2_score(y, y_pred) * 100
        #Log Matrics
        mlflow.log_metric({f"mse_{label}", mse,
                          f"mae_{label}", mae,
                          f"mape_{label}", mape,
                          f"r2_{label}", r2
                          })
        logger.info(f'==> Done processing evaluate() on {label}')
        return True
    except Exception as e:
        logger.error(f'in evaluate(): {e}')
        return False