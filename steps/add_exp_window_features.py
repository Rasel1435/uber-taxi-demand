import logging
import pandas as pd

from zenml import step
from typing import Union
from dask import dataframe as dd
from feature_engine.timeseries.forecasting import ExpandingWindowFeatures

from logs import configure_logger
logger = configure_logger()

@step(name='Generate exp Window Features', enable_step_logs=True, enable_artifact_metadata=True)
def ADDExpandingWindowFeatures(
    data: Union[pd.DataFrame, dd.DataFrame]) -> Union[pd.DataFrame, dd.DataFrame, None]:
    """
    Add expanding window features to the dataframe.
    """
    try:
        logger.info("==> Processing ADDExpandingWindowFeatures()")
        expwindow = ExpandingWindowFeatures(
            variables=None, min_periods=None, functions='std',
            periods=1, freq=None, sort_index=True,
            missing_values='raise', drop_original=False
        )
        expwindow.fit(data[['timestamp', 'passenger_demand', 'taxi_demand']])
        features = expwindow.fit_transform(
            data[['timestamp', 'passenger_demand', 'taxi_demand']])
        for col in list(features.columns)[3:]:
            data[col] = features[col].values
        logger.info(f'==> Successfully processed ADDExpandingWindowFeatures()')
        return data
    except Exception as e:
        logger.error(f'in ADDExpandingWindowFeatures(): {e}')
        return None