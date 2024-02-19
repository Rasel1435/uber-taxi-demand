from zenml import step
from dask import dataframe as dd
import logging
import pandas as pd
from typing import Union


logger = logging.getLogger(__name__)


def optimizeToFitMemory(ddf) -> bool:

    try:
        new_types = dict(
            int32=['passenger_count',],
            int16=['VendorID',]
        )
        for key in new_types:
            for col in new_types[key]:
                ddf[col] = ddf[col].astype(key)
        logger.info(f'==> Successfully processed OptimizeToFitMemory()')
        return ddf
    except Exception as e:
        return False


@step(name='Data Ingestion', enable_step_logs=True, enable_artifact_metadata=True)
def ingest_data(DATA_SOURCE) -> Union[pd.DataFrame, None]:
    """
    Get data from the source and return a DataFrame.
    """
    global optimizeToFitMemory
    try:
        logging.info(f'==> Processing ingest()')
        # Read the Parquet file directly into a Dask DataFrame
        ddf = dd.read_parquet(DATA_SOURCE, engine="pyarrow")
        # If you want to compute and save the Dask DataFrame as a single Parquet file
        # output_path = "yellow_tripdata_2023-10_dask.parquet"
        # df.to_parquet(output_path, engine="pyarrow", compression="snappy")
        start_of_month = DATA_SOURCE.split(
            '.parquet')[0][-7:] + '-01 00:00:00'
        s1 = ddf.loc[
            ddf.tpep_pickup_datetime >= start_of_month,
            ['tpep_pickup_datetime', 'passenger_count', 'VendorID']
        ]
        s2 = s1.set_index('tpep_pickup_datetime', sorted=True)
        s3 = s2.ffill()
        s4 = s3.resample('H').agg(
            {'passenger_count': 'sum', 'VendorID': 'count'})
        s5 = s4.map_partitions(optimizeToFitMemory)
        s6 = s5.reset_index()
        ddf = s6.compute()
        # If you want to compute and save the Dask DataFrame as multiple Parquet files (partitions)
        # output_path = "yellow_tripdata_2023-10_dask"
        # ddf.to_parquet(output_path, engine="pyarrow",
        #                compression="snappy", write_index=False)
        # ddf = ddf.head(ddf.shape[0])
        logger.info(f'==> Successfully processed ingest')
        return ddf
    except Exception as e:
        logger.error(f'in extract(): {e}')
        return None


if __name__ == '__main__':
    df = ingest_data(
        DATA_SOURCE='D:/uber-taxi-demand/data/yellow_tripdata_2022-01.parquet')
    print(df.head())
    print(df.shape)
