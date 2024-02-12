import pandas as pd

from zenml import step
from typing import Union
from dask import dataframe as dd
from sklearn.tree import DecisionTreeRegressor
from feature_engine.selection import SmartCorrelatedSelection, RecursiveFeatureElimination

from logs.logs import configure_logger
logger = configure_logger()
@step(enable_cache=True)
def SelectBestFeatures(
    data: Union[pd.DataFrame, dd.DataFrame]) -> Union[pd.DataFrame, dd.DataFrame, None]:
    """
    Select best features from the data
    """
    try:
        logger.info("==> Processing SelectBestFeatures()")
        X = data.drop(columns=['timestamp', 'passenger_demand', 'taxi_demand'])
        y = data['taxi_demand']
        timsestamp = data['timestamp']
        scs = SmartCorrelatedSelection(
            variables=None, method='pearson', threshold=0.5,
            missing_values='ignore', selection_method='variance',
            confirm_variables=False
        )
        scs_columns = set(scs.fit_transform(X).columns)
        rfe = RecursiveFeatureElimination(
            DecisionTreeRegressor(max_depth=3), scoring='r2', cv=3, threshold=0.01,
            variables=None, confirm_variables=False
        )
        rfe_columns = rfe.fit_transform(X, y)
        scs_columns.update(rfe_columns)
        data = data[['timestamp']+list(scs_columns)]
        data['taxi_demand'] = y
        logger.info(f'==> Successfully processed SelectBestFeatures()')
        return data
    except Exception as e:
        logger.error(f'in SelectBestFeatures(): {e}')
        return None