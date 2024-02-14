import warnings
warnings.filterwarnings('ignore')
import os
import sys
current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append(os.path.join(parent, 'steps'))
import logging
from steps.train import trainXGB
from steps.evaluate import evaluate
from steps.split import split
from zenml import pipeline

from logs import configure_logger
logger = configure_logger()

#Now you can import the module from the current directory 

@pipeline(enable_cache=False, name='trainPipeline', enable_step_logs=True)
def trainPipeline():
    """
    Pipeline to train the model
    """
    logger.info("Training Pipeline Started")
    try:
        data = split()
        is_trained = trainXGB(X_train=data['X_train'], y_train=data['y_train'])
        evaluate(X=data['X_train'], y=data['y_train'])
        evaluate(X=data['X_test'], y=data['y_test'])
        logger.info("Training Pipeline Successfully Done")
        
    except Exception as e:
        logger.error(f"==>Error in Training Pipeline(): {e}")

if __name__ == '__main__':
    run = trainPipeline()