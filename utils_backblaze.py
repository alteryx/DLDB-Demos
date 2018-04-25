import os
import dask.dataframe as dd
import featuretools as ft
import pandas as pd
import numpy as np
from tqdm import tqdm

# Download from here:
# https://www.backblaze.com/b2/hard-drive-test-data.html

def downsample(df, frac):
    if df['failure'].any():
        return df
    elif np.random.sample() < frac:
        return df
    else:
        return pd.DataFrame()


def load_data_as_dataframe(data_dir='data', csv_glob='*.csv',
                           nrows=None, negative_downsample_frac=0.01):
    df = dd.read_csv(os.path.join(data_dir, csv_glob),
                     assume_missing=True)
    df['date'] = dd.to_datetime(df['date'])
    if nrows is not None:
        df = df.head(nrows)
    else:
        df = df.compute()
    df['failure'] = df['failure'].map({0: False, 1: True}).astype(bool)
    # smart_9_raw is hard drive operational age in hours. Not possible to be over 10 years old
    df = df[df['smart_9_raw'] < 24*365.25*10]

    df = df.groupby('serial_number').apply(
        lambda x: downsample(x, negative_downsample_frac))
    return df


def load_entityset_from_dataframe(df):
    entityset = ft.EntitySet('BackBlaze')

    entityset.entity_from_dataframe("SMART_observations",
                                    df,
                                    index='id',
                                    make_index=True,
                                    time_index='date',
                                    variable_types={"failure": ft.variable_types.Boolean,
                                                    "smart_9_raw": ft.variable_types.Timedelta,
                                                    })

    entityset.normalize_entity(base_entity_id="SMART_observations",
                               new_entity_id="HDD",
                               index="serial_number",
                               additional_variables=['model',
                                                     'capacity_bytes'],
                               make_time_index=True,
                               new_entity_time_index='earliest_recording')
    entityset.normalize_entity(base_entity_id='HDD',
                               new_entity_id='models',
                               index='model',
                               make_time_index=False)
    entityset.add_last_time_indexes()
    return entityset


def create_labels(es, lead, min_training_data):
    tqdm.pandas(desc="Creating labels...")
    label_times = es['SMART_observations'].df.groupby('serial_number').progress_apply(
            lambda df: create_labels_per_instance(df, lead, min_training_data))
    label_times = (label_times.reset_index("serial_number")
                              .set_index("serial_number")
                              .set_index("cutoff",
                                         append=True))['label'].sort_index()
    return label_times.astype(bool)


def create_labels_per_instance(df, lead, min_training_data):
    start = df.iloc[0]['date']
    df = df[df['date'] > (start + min_training_data)]
    if df.empty:
        return pd.DataFrame()
    failure = df[df['failure']]
    if failure.empty:
        label_time = pd.DataFrame({"label": [False],
                                   "cutoff": [df["date"].sample(1).iloc[0]]})
    else:
        failure = failure.iloc[0]
        if failure['date'] - lead - min_training_data < start:
            return pd.DataFrame()
        label_time = pd.DataFrame({"label": [True],
                                   "cutoff": [failure["date"] - lead]})
    return label_time


def cutoff_raw_data(df, cutoffs, training_window):
    if isinstance(cutoffs, pd.Series):
        cutoffs = cutoffs.reset_index('cutoff').reset_index('serial_number')[['serial_number', 'cutoff']]
    else:
        cutoffs = cutoffs.copy()
    if not isinstance(training_window, pd.Timedelta):
        training_window = pd.Timedelta(training_window)
    cutoffs['start'] = cutoffs['cutoff'] - training_window
    merged = df.merge(cutoffs, on='serial_number', how='left')
    cutoff_data = merged[(merged['date'] <= merged['cutoff']) &
                         (merged['date'] >= merged['start'])]
    return (cutoff_data.drop(['cutoff', 'start'], axis=1)
                       .set_index('serial_number')
                       .set_index('date', append=True)
                       .sort_index())
