import pandas as pd

from zenml import step
from typing import Union
from dask import dataframe as dd
from sklearn.decomposition import PCA

from logs.logs import configure_logger
logger = configure_logger()

@step(enable_cache=True)
def ReduceDimensionality(
    data: Union[pd.DataFrame, dd.DataFrame]) -> Union[pd.DataFrame, dd.DataFrame, None]:
    """
    Reduce the dimensionality of the data by using PCA.
    """
    try:
        logger.info(f'==> Processing ReduceDimensionality()')
        features = data.drop(columns=['taxi_demand'])
        n_samples, n_features = features.shape
        target = data['taxi_demand']
        pca = PCA(n_components=3)
        features_reduced = pca.fit_transform(features)
        data = pd.DataFrame(features_reduced, columns=[
            f'PC{i}' for i in range(1, 4)])
        data['taxi_demand'] = target
        logger.info(f'==> Successfully processed ReduceDimensionality()')
        return data
    except Exception as e:
        logger.error(f"in ReduceDimensionality(): {e}")
        return None