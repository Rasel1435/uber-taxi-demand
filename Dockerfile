# Use official Python image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Copy requirements.txt and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ZenML Setup
RUN zenml integration install -y sklearn xgboost aws s3 mlflow
RUN zenml init
RUN zenml stack list
RUN zenml experiment-tracker register UberTaxiDemandTracker --flavor=mlflow
RUN zenml model-deployer register UberTaxiDemandModelDeployer --flavor=mlflow
RUN zenml stack register UberTaxiDemandDevStack -a default -o default -d UberTaxiDemandModelDeployer -e UberTaxiDemandTracker
RUN zenml init
RUN zenml stack set UberTaxiDemandDevStack

# Copy the entire project directory into the container
COPY . .

# Expose any ports the app is expecting
EXPOSE 8080
EXPOSE 5000
EXPOSE 8237
EXPOSE 8000

# RUN zenml up --port 8237
# RUN mlflow ui -p 5000
# Command to run the Time Series Forecasting project

