import config
import logging
import pandas as pd

from zenml import pipeline
from steps.split import split
from steps.train import train


@pipeline(enable_cache=True)
def run_train_pipeline():
    try:
        logging.info(f'==> Processing run_pipeline()')
        
        data = split()
        data = train(data) 
        
        logging.info(f'==> Successfully processed run_pipeline()')
    except Exception as e:
        logging.error(f'==> Error in run_pipeline(): {e}')
        
        
if __name__ == "__main__":
    run = run_train_pipeline()