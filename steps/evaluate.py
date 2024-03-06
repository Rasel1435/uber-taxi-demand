import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from zenml import step, client
from typing import Annotated, Tuple
from sklearn.base import BaseEstimator
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.metrics import mean_squared_error

from logs import configure_logger
logger = configure_logger()
    
tracker = client.Client().active_stack.experiment_tracker


def compute_aic_bic(y, y_pred, num_params):
    """
    Compute the AIC and BIC scores.

    Parameters:
    - y (array-like): Observed values.
    - y_pred (array-like): Predicted values.
    - num_params (int): Number of model parameters.

    Returns:
    - aic (float): AIC score.
    - bic (float): BIC score.
    """
    n = len(y)
    resid = y - y_pred
    rss = np.sum(resid ** 2)
    aic = 2 * num_params - 2 * np.log(rss)
    bic = n * np.log(rss / n) + num_params * np.log(n)
    return float(aic.values), float(bic.values)


@step(
    name='Evalution Step', experiment_tracker=tracker.name, enable_step_logs=True,
    enable_artifact_metadata=True, enable_artifact_visualization=True)

def evaluate_model(
    model: Annotated[BaseEstimator, 'trained Model'],
    X: pd.DataFrame,
    y: pd.DataFrame,
) -> Tuple[Annotated[float, 'R2 Score'], Annotated[float, 'MAPE']]:
    """
    Evaluate the model
    """
    try:
        logger.info("Evaluating model...")
        y_pred = model.predict(X).reshape(y.shape[0], 1)

        # MAPE, MSE, RMSE, R2, AIC, BIC
        mape = mean_absolute_percentage_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        rmse_ = float(rmse(y, y_pred))
        num_params = X.shape[1] + 2
        aic_, bic_ = compute_aic_bic(y, y_pred, num_params)
        mlflow.log_metrics(dict(mape=mape, mse=mse, r2=r2,
                           rmse=rmse_, aic=aic_, bic=bic_))
        return r2, mape
    except Exception as e:
        logger.error(e)
        raise e
