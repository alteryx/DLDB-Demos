import os
import dask.dataframe as dd
import featuretools as ft

# Download from here:
# https://www.backblaze.com/b2/hard-drive-test-data.html


def load_entityset(data_dir='data'):

    entityset = ft.EntitySet('BackBlaze')

    df = dd.read_csv(os.path.join(data_dir, '2017-01-0*.csv'),
                     assume_missing=True).compute()
    entityset.entity_from_dataframe("SMART_observations",
                                    df,
                                    index='id',
                                    make_index=True,
                                    time_index='date',
                                    variable_types={"failure":
                                                    ft.variable_types.Boolean})

    entityset.normalize_entity(base_entity_id="SMART_observations",
                               new_entity_id="HDD",
                               index="serial_number",
                               additional_variables=['model',
                                                     'capacity_bytes'],
                               make_time_index=True,
                               new_entity_time_index='timePM')
    return entityset
