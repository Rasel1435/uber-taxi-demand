import pandas as pd
import logging
from zenml import step
from typing import Annotated, Tuple
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@step(
    name="Split Data", enable_artifact_metadata=True,
    enable_artifact_visualization=True, enable_step_logs=True)
def split_data(
    data: Annotated[pd.DataFrame, 'features and Target'],
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[Annotated[pd.DataFrame, 'X_train'],
           Annotated[pd.DataFrame, 'X_test'],
           Annotated[pd.Series, 'y_train'],
           Annotated[pd.Series, 'y_test']]:
    """
    Split the data into train and test sets.
    """
    logger.info("Splitting the data into train and test sets.")
    try:
        logger.info(f'==> Processing split()')
        data = pd.read_parquet('data/feature.parquet')
        X = data.drop(columns=["taxi_demand", 'timestamp'])
        y = data.taxi_demand
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        logger.info(f'==> Successfully processed splitting()')
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f'in splitting(): {e}')
        raise e
