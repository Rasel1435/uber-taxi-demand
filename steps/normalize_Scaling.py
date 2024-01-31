import os
import config
import joblib
import logging
import pandas as pd

from zenml import step
from typing import Union
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@step(enable_cache=True)
def NormalizeScaling(data: pd.DataFrame) -> Union[pd.DataFrame, None]:
    """Scaling step.
    Args:
        data: Input data.
    Returns:
        Normalized data.
    """
    try:
        logger.info(f'==> Processing NormalizeScaling()')
        scaler = StandardScaler()
        # Assuming the data is a pandas DataFrame
        temp = data[['timestamp', 'taxi_demand']]
        data.drop(columns=['taxi_demand', 'timestamp'], inplace=True)
        scaler.fit(data)
        data = pd.concat(
            [temp, pd.DataFrame(scaler.transform(data), columns=data.columns)], axis=1)
        del temp
        # save Scaler model
        joblib.dump(scaler, os.path.join('model', 'scaler.pkl'))
        logger.info(f'Scaler model saved to {os.path.join("model", "scaler.pkl")}')
        print(data.columns)
        logger.info(f'==> Successfully processed NormalizeScaling()')
        return data
    except Exception as e:
        logger.error(f"in NormalizeScaling(): {e}")
        return None


# if __name__ == "__main__":
#     data = pd.read_csv("data/train.csv")
#     print(NormalizeScaling(data))