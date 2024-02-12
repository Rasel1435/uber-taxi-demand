from zenml import pipeline
from steps.split import split
from steps.train import train

from logs.logs import configure_logger
logger = configure_logger()


@pipeline(enable_cache=True)
def run_train_pipeline():
    try:
        logger.info(f'==> Processing run_pipeline()')
        
        data = split()
        data = train(data) 
        
        logger.info(f'==> Successfully processed run_pipeline()')
    except Exception as e:
        logger.error(f'==> Error in run_pipeline(): {e}')
        
        
if __name__ == "__main__":
    run = run_train_pipeline()