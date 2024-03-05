import pandas as pd
import dask.dataframe as dd
import hopsworks

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from feature_engine.datetime import DatetimeFeatures
from feature_engine.selection import SmartCorrelatedSelection, RecursiveFeatureElimination
from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures, ExpandingWindowFeatures
from math import sqrt

### Logging Configuration
# dZlKejE400BYPINW.93JR3IHAIDib8v4WliVyFSRttJXC8Jbgb40ZTWmrBN0ksAtKyoL0cHsZZ4Aj9BsM
from logs import configure_logger
# Call configure_logger function to set up the logger
logger = configure_logger()
PATH = '.'

# ETL - Extract


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


def extract(path: str) -> None:
    global ddf
    try:
        # Read the Parquet file directly into a Dask DataFrame
        ddf = dd.read_parquet(path, engine="pyarrow")

        # If you want to compute and save the Dask DataFrame as a single Parquet file
        # output_path = "yellow_tripdata_2023-10_dask.parquet"
        # df.to_parquet(output_path, engine="pyarrow", compression="snappy")

        start_of_month = path.split('.parquet')[0][-7:] + '-01 00:00:00'
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
        logger.info(f'==> Successfully processed extract()')
        return True
    except Exception as e:
        logger.error(f'in extract(): {e}')
        return False

# ETL - Transform


def clean() -> bool:
    global ddf
    try:

        ddf['timestamp'] = pd.to_datetime(ddf.tpep_pickup_datetime)
        ddf.drop(columns=['tpep_pickup_datetime'], inplace=True)
        ddf.rename(
            {
                'passenger_count': 'passenger_demand', 'VendorID': 'taxi_demand'
            }, axis=1, inplace=True
        )
        ddf.drop_duplicates(subset=['timestamp'], inplace=True)
        logger.info(f'==> Successfully processed clean()')
        return True
    except Exception as e:
        logger.error(f'in clean(): {e}')
        return False
    
        
# featureEngineering start here
def add_temporal_features() -> None:
    global ddf
    try:
        features_to_extract = [
            "month", "quarter", "semester", "year", "week", "day_of_week", "day_of_month",
            "day_of_year", "weekend", "month_start", "month_end", "quarter_start",
            "quarter_end", "year_start", "year_end", "leap_year", "days_in_month", "hour", "minute", "second"
        ]
        temporal = DatetimeFeatures(
            features_to_extract=features_to_extract).fit_transform(ddf[['timestamp']])
        for col in temporal.columns:
            ddf.loc[:, col] = temporal[col].values
        logger.info(f'==> Successfully processed add_temporal_features()')
    except Exception as e:
        logger.error(f'in add_temporal_features(): {e}')
        

def add_lag_features() -> None:
    global ddf
    try:
        lagfeatures = LagFeatures(variables=None, periods=[1, 2, 4, 8, 16, 24], freq=None, sort_index=True,
                                  missing_values='raise', drop_original=False)
        lagfeatures.fit(ddf[['timestamp', 'passenger_demand', 'taxi_demand']])
        features = lagfeatures.transform(
            ddf[['timestamp', 'passenger_demand', 'taxi_demand']])
        for col in list(features.columns)[3:]:
            ddf[col] = features[col].values
        logger.info(f'==> Successfully processed add_lag_features()')
    except Exception as e:
        logger.error(f'in The add_lag_features(): {e}')
        
        
def add_window_features() -> None:
    global ddf
    try:
        window = WindowFeatures(
            variables=None, window=7, min_periods=1,
            functions=['mean', 'std', 'median'], periods=1, freq=None, sort_index=True,
            missing_values='raise', drop_original=False
        )
        window.fit(ddf[['timestamp', 'passenger_demand', 'taxi_demand']])
        features = window.fit_transform(
            ddf[['timestamp', 'passenger_demand', 'taxi_demand']])
        for col in list(features.columns)[3:]:
            ddf[col] = features[col].values
        logger.info(f'==> Successfully processed add_window_features()')
    except Exception as e:
        logger.error(f'in add_window_features(): {e}')

        
        
def add_exp_window_features() -> None:
    global ddf
    try:
        expwindow = ExpandingWindowFeatures(
            variables=None, min_periods=None, functions='std',
            periods=1, freq=None, sort_index=True,
            missing_values='raise', drop_original=False
        )
        expwindow.fit(ddf[['timestamp', 'passenger_demand', 'taxi_demand']])
        features = expwindow.fit_transform(
            ddf[['timestamp', 'passenger_demand', 'taxi_demand']])
        for col in list(features.columns)[3:]:
            ddf[col] = features[col].values
        logger.info(f'==> Successfully processed add_exp_window_features()')
    except Exception as e:
        logger.error(f'in add_exp_window_features(): {e}')
        
        
# Feture Selection Start here
def select_best_features():
    global ddf, timestamp
    try:
        X = ddf.drop(columns=['timestamp', 'passenger_demand', 'taxi_demand'])
        y = ddf['taxi_demand']
        timsestamp = ddf['timestamp']
        scs = SmartCorrelatedSelection(
            variables=None, method='pearson', threshold=0.5,
            missing_values='ignore', selection_method='variance',
            confirm_variables=False
        )
        scs_columns = set(scs.fit_transform(X).columns)
        rfe = RecursiveFeatureElimination(
            DecisionTreeRegressor(max_depth=3), scoring='r2', cv=3, threshold=0.01,
            variables=None, confirm_variables=False
        )
        rfe_columns = rfe.fit_transform(X, y)
        scs_columns.update(rfe_columns)
        ddf = ddf[['timestamp']+list(scs_columns)]
        ddf['taxi_demand'] = y
        logger.info(f'==> Successfully processed select_best_features()')
    except Exception as e:
        logger.error(f'in select_best_features(): {e}')
        

# Data Scaling here
def normalizeScaling() -> None:
    global ddf
    try:
        scaler = StandardScaler()
        scaler.fit(ddf.drop(columns=['taxi_demand',]))
        ddf.loc[:, ddf.columns[:-1]
                ] = scaler.transform(ddf.drop(columns=['taxi_demand',]))
        logger.info(f'==> Successfully processed normalizeScaling()')
    except Exception as e:
        logger.error(f"in normalizeScaling(): {e}")
        
        
# DimensonalRedaction start from here
def reduceDimensionality() -> None:
    global ddf
    try:
        features = ddf.drop(columns=['taxi_demand'])
        n_samples, n_features = features.shape
        target = ddf['taxi_demand']
        pca = PCA(n_components=3)
        features_reduced = pca.fit_transform(features)
        ddf = pd.DataFrame(features_reduced, columns=[
            f'PC{i}' for i in range(1, 4)])
        ddf['taxi_demand'] = target
        logger.info(f'==> Successfully processed reduceDimensionality()')
    except Exception as e:
        logger.error(f"in reduceDimensionality(): {e}")
        
        
# Now Time to call the all function and save it 
def transform():
    global ddf
    try:
        clean()
        add_temporal_features()
        add_lag_features()
        add_window_features()
        add_exp_window_features()
        ddf.dropna(inplace=True)
        if ddf is None or ddf.empty:
            raise ValueError(
                "DataFrame is None or empty start_of_month dropping missing values.")
        # call other steps
        select_best_features()
        normalizeScaling()
        reduceDimensionality()
        ddf.dropna(inplace=True)
        logger.info(f'==> Successfully processed transform()')

    except Exception as e:
        logger.error(f'in preprocessFeatures(): {e}')
        
        
def load_features(featuregroup_name, dataframe):
    """
    Load features into a feature group in Hopsworks Feature Store.

    Parameters:
    - featurestore_name (str): Name of the feature store in Hopsworks.
    - featuregroup_name (str): Name of the feature group to load the features into.
    - dataframe (pandas.DataFrame): DataFrame containing the features.

    Returns:
    - None
    """
    try:
        # Connect to the feature store
        project = hopsworks.login(
            api_key_value="dZlKejE400BYPINW.93JR3IHAIDib8v4WliVyFSRttJXC8Jbgb40ZTWmrBN0ksAtKyoL0cHsZZ4Aj9BsM")
        fs = project.get_feature_store()
        featurestore = fs.get_or_create_feature_group(
            name=featuregroup_name, version=1)
        # Insert the features into the feature group
        featurestore.insert(dataframe, write_options={"wait_for_job": True},)
        logger.info(f'==> Successfully processed load_features()')

    except Exception as e:
        logger.error(f'in load_features(): {e}')


if __name__ == '__main__':
    # Dictionary to store processed data
    ddf = None

    extract(r"C:/Users/SRA/Desktop/backup/C/MLgrit/time_series_project/uber-taxi-demand/data/2022/january_2022-01.parquet")
    # extract(
    #     r"https://d37ci6vzurychx.cloudfront.net/uber-taxi-demand/data/2023/january_2023-01.parquet"
    # )
    # transform()
    # Set your feature store and feature group names
    # featurestore_name = "your_featurestore"
    featuregroup_name = "taxi_demand"

    # Load your features into a Pandas DataFrame (replace this with your actual data loading logic)
    # For example, you can use Pandas read_csv or any other method to load your data
    # dataframe = pd.read_csv("your_data.csv")

    # Assuming you have a Pandas DataFrame 'dataframe' with your features
    # load_features(featuregroup_name, ddf)
