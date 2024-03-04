from zenml import step
from zenml.steps import BaseParameters, Output
from pydantic import BaseModel
from typing import Annotated
import logging


# Create Deployment Trigger
class DeploymentTrigger(BaseModel):
    """It will trigger the deployment"""
    min_accuracy : float = 0.92


@step(name='DeploymentTrigger', enable_step_logs=True, enable_artifact_metadata=True)
def trigger_deployment(accuracy: Annotated[float, 'accuracy'] ,deployment_trigger: Annotated[DeploymentTrigger, 'deployment_trigger']) -> Annotated[bool, 'decision']:
    """It will trigger the deployment"""
    decision = accuracy >= deployment_trigger.min_accuracy 
    if decision:
        logging.info(f"Triggering the deployment with min_accuracy {deployment_trigger.min_accuracy}")
        return decision
    return decision