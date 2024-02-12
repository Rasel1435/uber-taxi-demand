import logging
import pandas as pd

from zenml import step
from dask import dataframe as dd
from typing import Union
from feature_engine.timeseries.forecasting import WindowFeatures

from logs.logs import configure_logger
logger = configure_logger()


@step(enable_cache=True)
def AddWindowFeatures(
        data: Union[dd.DataFrame, pd.DataFrame]) -> Union[dd.DataFrame, pd.DataFrame, None]:
    """Add window features to the dataframe

    Args:
        data (Union[dd.DataFrame, pd.DataFrame]): The dataframe to add window features to.

    Returns:
        Union[dd.DataFrame, pd.DataFrame]: The dataframe with window features added.
    """

    try:
        logger.info(f"==> Processing AddWindowFeatures()")
        windowfeatures = WindowFeatures(variables=None, window=24, freq=None, sort_index=True,
                                        missing_values='raise', drop_original=False)
        windowfeatures.fit(
            data[['timestamp', 'passenger_demand', 'taxi_demand']])
        features = windowfeatures.transform(
            data[['timestamp', 'passenger_demand', 'taxi_demand']])
        for col in list(features.columns)[3:]:
            data[col] = features[col].values
        logger.info(f'==> Successfully processed AddWindowFeatures()')
        return data
    except Exception as e:
        logger.error(f'in AddWindowFeatures(): {e}')
        return None