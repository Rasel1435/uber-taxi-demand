import logging
import pandas as pd

from zenml import step
from typing import Union
from dask import dataframe as dd


logger = logging.getLogger(__name__)


@step(enable_cache=True)
def clean_data(data: Union[pd.DataFrame, dd.DataFrame]) -> Union[pd.DataFrame, dd.DataFrame, None]:
    """
    Clean the data by removing duplicates and null values.
    """

    try:
        logger.info("==> Processing clean_data()")
        data['timestamp'] = pd.to_datetime(data.tpep_pickup_datetime)
        data.drop(columns=['tpep_pickup_datetime'], inplace=True)
        data.rename(
            {
                'passenger_count': 'passenger_demand', 'VendorID': 'taxi_demand'
            }, axis=1, inplace=True
        )
        data.drop_duplicates(subset=['timestamp'], inplace=True)
        logger.info(f'==> Successfully processed clean_data()')
        return data
    except Exception as e:
        logger.error(f'in clean_data(): {e}')
        return None
