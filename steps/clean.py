from zenml import step
from dask import dataframe as dd
import logging
import pandas as pd
from typing import Union


logger = logging.getLogger(__name__)


@step(name='Data Cleaning', enable_step_logs=True, enable_artifact_metadata=True)
def clean_data(data: Union[pd.DataFrame, dd.DataFrame]) -> Union[pd.DataFrame, dd.DataFrame, None]:
    """
    Clean the data by removing duplicates and null values.
    """

    try:
        logger.info("==> Cleaning data...")
        data = data.drop_duplicates()
        data = data.dropna(axis=0, how='any')
        data['timestamp'] = pd.to_datetime(data.tpep_pickup_datetime)
        data.drop(columns=['tpep_pickup_datetime'], inplace=True)
        data.rename(
            {
                'passenger_count': 'passenger_demand', 'VendorID': 'taxi_demand'
            }, axis=1, inplace=True
        )
        data.drop_duplicates(subset=['timestamp'], inplace=True)
        logger.info(f'==> Successfully processed clean()')
        return data
    except Exception as e:
        logger.error(f'in clean(): {e}')
        return None
