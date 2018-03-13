
# coding: utf-8

# In[ ]:


import featuretools as ft
import pandas as pd
import numpy as np
from dask import bag
from dask.diagnostics import ProgressBar
import utils_instacart as utils
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from dldb import DLDB, tdfs, make_temporal_cutoffs
from keras.callbacks import EarlyStopping
import os
ft.__version__


# In[ ]:


# pbar = ProgressBar()
# pbar.register()


# # In[ ]:


# path = "partitioned_data/"
# _, dirnames, _ = next(os.walk(path))
# dirnames = [path+d for d in dirnames]
# b = bag.from_sequence(dirnames)
# entity_sets = b.map(utils.load_entityset)


# # In[ ]:


# cutoff_time = pd.Timestamp('March 1, 2015')
# training_window = ft.Timedelta("60 days")


# # In[ ]:


# label_times = entity_sets.map(utils.make_labels,
                              # product_name="Banana",
                              # cutoff_time=cutoff_time,
                              # prediction_window=ft.Timedelta("4 weeks"),
                              # training_window=training_window)


# # In[ ]:


# tdfs_results = entity_sets.map(tdfs,
                               # target_entity="users",
                               # cutoffs=label_times,
                               # training_window=training_window,
                               # window_size='14d',
                               # num_windows=5,
                               # verbose=True)


# # In[ ]:


# label_times, tdfs_results = bag.compute(label_times, tdfs_results)

# labels = pd.concat(label_times).set_index('user_id').sort_index()['label']
# fm = pd.concat([r[0] for r in tdfs_results]).sort_index()
# fl = tdfs_results[0][1]


# # In[ ]:


#fm.to_csv("fm_full_data.csv")
#labels.to_frame().to_csv("label_times_full_data.csv")
fm = pd.read_csv("fm_full_data.csv", index_col=["user_id", "time"], parse_dates=["time"])
labels = pd.read_csv("label_times_full_data.csv", index_col=["user_id"])["label"]
es = utils.load_entityset('partitioned_data/part_0/')
fl = ft.load_features('fl.p', es)

print("loaded")
# In[ ]:


dl_model = DLDB(
    regression=False,
    classes=[False, True],
    recurrent_layer_sizes=(32, 32),
    dense_layer_sizes=(32, 32),
    dropout_fraction=0.2,
    recurrent_dropout_fraction=0.1,
    categorical_embedding_size=20,
    categorical_max_vocab=12)
# TODO: cheating a bit, put back in CV later
dl_model.compile(fm, fl=fl)
print("compiled")


# In[ ]:


cv_score = []
n_splits = 3
test_frac = 0.1
# Use 10% of data as testing set, but only run 3 rounds of cross-validation
# (because they take a while)
splitter = StratifiedKFold(n_splits=int(1/test_frac), shuffle=True)

for i, train_test_index in enumerate(splitter.split(labels, labels)):
    train_labels = labels.iloc[train_test_index[0]]
    test_labels = labels.iloc[train_test_index[1]]
    train_fm = fm.loc[(train_labels.index, slice(None)), :]
    test_fm = fm.loc[(test_labels.index, slice(None)), :]

    dl_model.fit(
        train_fm, train_labels,
        validation_split=0.1,
        epochs=100,
        batch_size=32,
        callbacks=[EarlyStopping()])

    predictions = dl_model.predict(test_fm)
    cv_score.append(roc_auc_score(test_labels, predictions))
    if i == n_splits - 1:
        break
mean_score = np.mean(cv_score)
stderr = 2 * (np.std(cv_score) / np.sqrt(n_splits))

print("AUC %.2f +/- %.2f" % (mean_score, stderr))

