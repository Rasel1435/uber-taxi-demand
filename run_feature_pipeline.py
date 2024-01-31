import config
import logging

from zenml import pipeline
from steps.ingest import ingest_data
from steps.clean import clean_data
from steps.add_temporal_features import AddTemporalFeatures
from steps.add_lag_features import AddLagFeatures
from steps.add_window_features import AddWindowFeatures
from steps.add_exp_window_features import ADDExpandingWindowFeatures
from steps.select_best_features import SelectBestFeatures
from steps.normalize_Scaling import NormalizeScaling
from steps.reduce_Dimensionality import ReduceDimensionality


@pipeline(enable_cache=True)
def run_pipeline():
    """
    Pipeline that runs the ingest, clean, lag and window features.
    """
    try:
        logging.info(f'==> Processing run_pipeline()')
        data = ingest_data(DATA_SOURCE=config.DATA_SOURCE)
        data = clean_data(data)
        data = AddTemporalFeatures(data)
        data = AddLagFeatures(data)
        data = AddWindowFeatures(data)
        data = ADDExpandingWindowFeatures(data)
        data = SelectBestFeatures(data)
        data = NormalizeScaling(data)
        # data = ReduceDimensionality(data)
        logging.info(f'==> Successfully processed run_pipeline()')
    except Exception as e:
        logging.error(f'==> Error in run_pipeline(): {e}')


if __name__ == "__main__":
    run = run_pipeline()