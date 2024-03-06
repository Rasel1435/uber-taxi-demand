import click, config

from typing import cast, Annotated
from pipelines.deploy import continuous_deployment
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer


def deploy_model(
        min_accuracy: Annotated[float, 'min_accuracy'] = 0.92, 
        workers: Annotated[int, 'workers'] = 1, 
        timeout: Annotated[int, 'timeout'] = 90
    ) -> None:
    """Deploy model to production."""
    click.echo("Deploying model to production")
    try:
        deployer = MLFlowModelDeployer.get_active_model_deployer()
        deployment_pipeline = continuous_deployment(min_accuracy=min_accuracy, workers=workers, timeout=timeout)
        # fetch existing services with same pipeline name, step name and model name
        existing_services = deployer.find_model_server(
            pipeline_name="continuous_deployment",
            pipeline_step_name="mlflow_model_deployer_step",
            model_name=f'{config.MODEL_NAME}-XGBoost',
        )
        if existing_services:
            service = cast(MLFlowDeploymentService, existing_services[0])
            if service.is_running:
                print(
                    f"The MLflow prediction server is running locally as a daemon",
                    f"process service and accepts inference requests at: \n",
                    f" {service.prediction_url}\n",
                    f"To stop the service, run 11",
                    f" [italic green] zenml model-deployer models delete ",
                    f"{ str(service. uuid)}` [/italic green].",
                )
            elif service.is_failed:
                print(
                    f"The MLflow prediction server is in a failed state: \n",
                    f" Last state: '{service.status.state.value}'\n",
                    f" Last error: '{service.status. last_error}"
                )
            else:
                print(
                    "No MLflow prediction server 1is , currently running. The deployment 11",
                    "pipeline must run first to train a model and deploy it. Execute ",
                    "the same command with the `--deploy` argument to deploy a model.",
                )
        click.echo("Model deployed successfully")
    except Exception as e:
        click.echo(e)
        raise e


if __name__ == "__main__":
    deploy_model(min_accuracy=0.82, workers=2, timeout=90)
    # continuous_deployment(min_accuracy=0.82, workers=2, timeout=90)
    # deploy_model()
    # mlflow_model_deployer_step(model_name="model")
    # mlflow_model_server_step(model_name="model")
    # mlflow_deployment_step(model_name="model")
    # mlflow_deployment_service_step(model_name="model")
    # mlflow_deployment_service_list_step(model_name="model")
    # mlflow_deployment_service_filter_step(model_name="model")
    # mlflow_deployment_service_delete_step(model_name="model")
    # mlflow_deployment_service_stop_step(model_name="model")
    # mlflow_deployment_service_start_step(model_name="model")
    # mlflow_deployment_service_restart_step(model_name="model")
    # mlflow_deployment_service_status_step(model_name="model")