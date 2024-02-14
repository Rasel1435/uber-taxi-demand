import sys
import os
current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append(os.path.join(parent, 'steps'))

from steps.train import trainXGB
from steps.split import split
from steps.evaluate import evaluate
from zenml import pipeline, client

from logs import configure_logger
logger = configure_logger()

# Now you can import modules from the parent directory
@pipeline(enable_cache=True, enable_artifact_metadata=True,
          name='trainPipelineUberTaxiDemand',enable_step_logs=True)
def trainPipeline():
    """
    Pipeline trains Model.
    """
    try:
        logger.info('==> Starting trainPipeline()')
        
        data = split()
        model = trainXGB(data)

        evaluate(data=data, model=model, label='TRAIN')
        evaluate(data=data, model=model, label='TEST')
        
        logger.info('==> trainPipeline() Completed')

    except Exception as e:
        logger.error(f'==> Error in trainPipeline(): {e}')


if __name__ == "__main__":
    run = trainPipeline()
    print(f'Track Experiment Here => {client.Client().active_stack.experiment_tracker.get_tracking_uri()}')
#ubertaxidemandartifactstore
# zenml artifact-store register UberTaxiDemandArtifactStore --flavor=s3     --path=s3://ubertaxidemandartifactstore --client_kwargs='{"endpoint_url": "http://my-s3-endpoint"}'