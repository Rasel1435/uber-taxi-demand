# Load Data of Previous Month & Current Month & both have to merge
# Preprocess the merge data - apply feature pipeline except feature selection
# Load Model & Predict 
#

import pandas as pd
from logs import configure_logger
logger = configure_logger()


feature_path = '../data/feature-2023.parquet'
model_path = ''
feature = pd.DataFrame()

def extract() -> None:
    global feature_path
    try:
        feature = pd.read_parquet(feature_path).tail(1)
    except Exception as e:
        print(e)


def getModel():
    pass