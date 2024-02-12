import pandas as pd

from zenml import step
from typing import Union
from feature_engine.datetime import DatetimeFeatures

from logs.logs import configure_logger
logger = configure_logger()


@step(enable_cache=True)
def AddTemporalFeatures(data: pd.DataFrame) -> Union[pd.DataFrame, None]:
    features_to_extract = [
        "month", "quarter", "semester", "week", "day_of_week", "day_of_month",
        "day_of_year", "weekend", "month_start", "month_end", "quarter_start",
        "quarter_end", "year_start", "year_end", "hour"
    ]

    try:
        logger.info(f'==> Processing AddTemporalFeatures()')
        temporal = DatetimeFeatures(
            features_to_extract=features_to_extract).fit_transform(data[['timestamp']])
        for col in temporal.columns:
            data.loc[:, col] = temporal[col].values
        logger.info(f'==> Successfully processed AddTemporalFeatures()')
        return data
    except Exception as e:
        logger.error(f'==> Error in AddTemporalFeatures()')
        return None