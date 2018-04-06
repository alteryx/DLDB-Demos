import pandas as pd

import featuretools as ft
from featuretools.selection import remove_low_information_features

from .load_entityset import load_entityset
from .. import save_features


def test(data_dir='data'):
    es = load_entityset(data_dir)

    cutoff_times = pd.DataFrame(
        {'serial_number': pd.Series(es['HDD'].get_all_instances()).sample(10, random_state=1)})
    cutoff_times['time'] = pd.Timestamp('1/10/2017')
    feature_matrix, features = ft.dfs(target_entity="HDD",
                                      entityset=es,
                                      cutoff_time=cutoff_times,
                                      verbose=True)
    save_features(__file__, feature_matrix, features)


def test_approx(data_dir='data'):
    es = load_entityset(data_dir)

    cutoff_times = pd.DataFrame(
        {'serial_number': pd.Series(
            es['HDD'].get_all_instances()).sample(10, random_state=1)})
    cutoff_times.reset_index(drop=True, inplace=True)
    cutoff_times['time'] = pd.Timestamp('1/10/2017')
    cutoff_times.loc[slice(0, 5), 'time'] = pd.Timestamp('1/9/2017')
    feature_matrix, features = ft.dfs(target_entity="HDD",
                                      entityset=es,
                                      approximate='7 days',
                                      cutoff_time=cutoff_times,
                                      verbose=True)
    save_features(__file__, feature_matrix, features, approx=True)


if __name__ == '__main__':
    test()
