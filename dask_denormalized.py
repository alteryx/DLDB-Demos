
# coding: utf-8

# In[ ]:


import featuretools as ft
from dask import bag
from dask.diagnostics import ProgressBar
import pandas as pd
import numpy as np
import utils_instacart as utils
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from dldb import DLDB
import os
ft.__version__


# In[ ]:


#pbar = ProgressBar()
#pbar.register()
#
#
## In[ ]:
#
#
#path = "partitioned_data/"
#_, dirnames, _ = next(os.walk(path))
#dirnames = [path+d for d in dirnames]
#b = bag.from_sequence(dirnames)
#entity_sets = b.map(utils.load_entityset)
#
#
## In[ ]:
#
#
#cutoff_time = pd.Timestamp('March 1, 2015')
#training_window = ft.Timedelta("60 days")
#
#
## In[ ]:
#
#
#label_times = entity_sets.map(utils.make_labels,
#                              product_name="Banana",
#                              cutoff_time=cutoff_time,
#                              prediction_window=ft.Timedelta("4 weeks"),
#                              training_window=training_window)
#
#
## In[ ]:
#
#
#denormed = entity_sets.map(utils.denormalize_entityset,
#                           cutoff_time=cutoff_time,
#                           training_window=training_window)
#
#
## In[ ]:
#
#
#label_times, denormed = bag.compute(label_times, denormed)
#labels = pd.concat(label_times).set_index('user_id').sort_index()['label']
#fm = pd.concat(denormed).sort_index()
fm = pd.read_csv("fm_denormed.csv", parse_dates=['order_time'], index_col=['user_id', 'order_time'])
fm.drop(['order_product_id'], axis=1, inplace=True)
labels = pd.read_csv("label_times_full_data.csv", index_col=["user_id"])["label"]
fm.reset_index('order_time', drop=True, inplace=True)



# In[ ]:


dl_model = DLDB(
    regression=False,
    classes=[False, True],
    recurrent_layer_sizes=(128, 128, 64),
    dense_layer_sizes=(64, 32),
    dropout_fraction=0.2,
    recurrent_dropout_fraction=0.2,
    categorical_embedding_size=64,
    categorical_max_vocab=None)
# TODO: cheating a bit, put back in CV later
dl_model.compile(fm, categorical_feature_names=[c for c in fm.columns if c != 'reordered'])


# In[ ]:


cv_score = []
n_splits = 10
# Use 10% of data as testing set, but only run 3 rounds of cross-validation
# (because they take a while)
splitter = StratifiedKFold(n_splits=n_splits, shuffle=True)

for i, train_test_index in enumerate(splitter.split(labels, labels)):
    train_labels = labels.iloc[train_test_index[0]]
    test_labels = labels.iloc[train_test_index[1]]
    train_fm = fm.loc[train_labels.index, :]
    test_fm = fm.loc[test_labels.index, :]

    dl_model.fit(
        train_fm, train_labels,
        validation_split=0.1,
        epochs=10,
        batch_size=256)

    predictions = dl_model.predict(test_fm)
    cv_score.append(roc_auc_score(test_labels, predictions))
mean_score = np.mean(cv_score)
stderr = 2 * (np.std(cv_score) / np.sqrt(n_splits))

print("AUC %.2f +/- %.2f" % (mean_score, stderr))
