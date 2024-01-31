import logging
import pandas as pd
import config

from zenml import step
from typing import Union
from dask import dataframe as dd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@step(enable_cache=True)
def NormalizeScaling(
    data: Union[pd.DataFrame, dd.DataFrame]) -> Union[pd.DataFrame, dd.DataFrame, None]:
    """Normalize scaling step.
    Args:
        data: Input data.
    Returns:
        Normalized data.
    """
    try:
        logger.info(f'==> Processing NormalizeScaling()')
        scaler = StandardScaler()
        # Assuming the data is a pandas DataFrame
        scaler.fit(data.drop(columns=['taxi_demand',]))
        data.loc[:, data.columns[:-1]
                ] = scaler.transform(data.drop(columns=['taxi_demand',]))
        logger.info(f'==> Successfully processed normalizeScaling()')
        return data
    except Exception as e:
        logger.error(f"in normalizeScaling(): {e}")
        return None
