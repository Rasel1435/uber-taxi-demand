import config
import logging
import hopsworks
import pandas as pd

from os import path
from zenml import step
from joblib import dump
from typing import Union


@step(enable_cache=True)
def load_features(data: pd.DataFrame) -> bool:
    """
    Load features into a feature group in Hopsworks Feature Store.

    Parameters:
    - data (pandas.DataFrame): DataFrame containing the features.

    Returns:
    - None
    """
    try:
        logging.info(
            f'==> Loading features into feature group {config.FEATURE_GROUP_NAME}')
        # Load the features into a feature group in Hopsworks Feature Store
        # Connect to the feature store
        project = hopsworks.login(api_key_value=config.API_KEY_HOPSWORKS)
        fs = project.get_feature_store()
        featurestore = fs.get_or_create_feature_group(
            name=config.FEATURE_GROUP_NAME, version=1, description=config.FEATURE_GROUP_DESCRIPTION,)
        # Insert the features into the feature group
        featurestore.insert(data, write_options={"wait_for_job": True},)
        # update feature descriptions
        for feature in config.FEATURE_DESCRIPTIONS:
            featurestore.update_feature_description(feature)
        # closing hopsworks connection
        project.close()
        logging.info(
            f'==> Successfully loaded features into feature group {config.FEATURE_GROUP_NAME}')
        return True
    except Exception as e:
        logging.error(
            f'==> Failed to load features into feature group {config.FEATURE_GROUP_NAME}')
        return False

if __name__ == "__main__":
    data = pd.read_parquet(config.DATA_SOURCE)
    load_features(data)