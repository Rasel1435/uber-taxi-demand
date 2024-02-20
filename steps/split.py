import pandas as pd

from zenml import step
from typing import Union, Dict
from sklearn.model_selection import train_test_split
from logs import configure_logger

logger = configure_logger()

@step(enable_cache=True)
def split() -> Union[Dict, None]:
    """
    Split the data into train and test sets.
    """
    logger.info("Splitting the data into train and test sets.")
    try:
        logger.info(f'==> Processing split()')
        data = pd.read_parquet('data/feature-2022.parquet')
        X = data.drop(columns=["taxi_demand",'timestamp'])
        y = data.taxi_demand
        X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
        
        logger.info(f'==> Successfully processed splitting()')
        return dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    
    except Exception as e:
        logger.error(f'in splitting(): {e}')
        return None