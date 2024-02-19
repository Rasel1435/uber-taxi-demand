import logging
from zenml import step
import pandas as pd
from typing import Union
from joblib import dump
import config
from os import path
import boto3
from boto3.dynamodb.types import TypeSerializer
from decimal import Decimal


@step(enable_cache=True)
def load_features(data: pd.DataFrame) -> bool:
    """
    Load features into a feature in DynamoDB Feature Store.

    Parameters:
    - data (pandas.DataFrame): DataFrame containing the features.

    Returns:
    - None
    """
    try:
        logging.info(
            f'==> Loading features into DynamoDB feature group {config.FEATURE_GROUP_NAME}')
        # Load the features into DynamoDB
        # Initialize DynamoDB client with credentials
        dynamodb = boto3.resource(
            'dynamodb', region_name='us-east-1',
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
        )

        # Get reference to the DynamoDB table
        table = dynamodb.Table(config.TABLE_NAME)

        # Convert DataFrame timestamps to strings
        data['timestamp'] = data['timestamp'].astype(str)
        print(data.isnull().sum())
        # Convert DataFrame floats to DynamoDB-compatible decimals
        for column in data.select_dtypes(include=['float64']).columns:
            data[column] = data[column].apply(lambda x: Decimal(str(x)))

        # Convert DataFrame to list of dictionaries
        data_to_append = data.head().to_dict(orient='records')

        # Batch write data to DynamoDB table
        for item in data_to_append:
            # Convert timestamp strings back to DynamoDB-compatible format
            item['timestamp'] = {'S': item['timestamp']}
            table.put_item(Item=item)
        logging.info("Loaded Into DynamoDB Successfully")
        return True
    except Exception as e:
        logging.error(
            f'==> Failed to load features into feature group {config.FEATURE_GROUP_NAME}: {e}')
        return False


if __name__ == "__main__":
    data = pd.read_csv(config.DATA_SOURCE)
    load_features(data)
