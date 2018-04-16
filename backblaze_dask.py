import featuretools as ft
import pandas as pd
import numpy as np
import utils_backblaze as utils
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from dldb import DLDB
import os
ft.__version__


data_dir = '/Users/bschreck/Google Drive File Stream/My Drive/Feature Labs Shared/EntitySets/entitysets/backblaze_harddrive/data'
#data_dir = '../backblaze_harddrive_data'

# df = utils.load_data_as_dataframe(data_dir=data_dir, csv_glob='*.csv')
df = utils.load_data_as_dataframe(data_dir=data_dir, csv_glob='*.csv', nrows=10)
print("loaded df")


# def upsample(df):
    # if df['failure'].any():
        # return df
    # elif np.random.sample() < 0.01:
        # return df
    # else:
        # return pd.DataFrame()

# df = df.groupby('serial_number').apply(upsample)
# print("upsampled df")
es = utils.load_entityset_from_dataframe(df)
print("loaded es")

training_window = "20 days"
lead = pd.Timedelta('1 day')
prediction_window = pd.Timedelta('25 days')
min_training_data = pd.Timedelta('5 days')


# labels = utils.create_labels(es,
                             # lead,
                             # min_training_data)
# fm = pd.read_csv('backblaze_ftens_high_info.csv', parse_dates=['time'], index_col=['serial_number', 'time']).sort_index()
# labels = pd.read_csv('backblaze_labels.csv', parse_dates=['cutoff'], index_col=['serial_number', 'cutoff'])['label'].sort_index()
#fl = ft.load_features('backblaze_high_info_fl.p', es)
fl = ft.load_features("fl_backblaze_selected.p", es)
fm = fm[f.get_name() for f in fl]
fm.to_csv("backblaze_ftens_selected.csv")

print("loaded labels")

# cutoffs = labels.reset_index('cutoff').reset_index('serial_number')[['serial_number', 'cutoff']]
# cutoff_raw = utils.cutoff_raw_data(df, cutoffs, training_window)
print("cutoff raw data")

dl_model = DLDB(
    regression=False,
    classes=[False, True],
    recurrent_layer_sizes=(32, 32),
    dense_layer_sizes=(32, 16),
    dropout_fraction=0.2,
    recurrent_dropout_fraction=0.2,
    categorical_embedding_size=12,
    categorical_max_vocab=20)


n_splits=20
splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

'''
DFS SCORES:
[0.6195353582921278,
 0.6610391649007141,
 0.6698341686568671,
 0.749285209727875,
 0.781316223216388,
 0.7547428436879814,
 0.6718692186080931,
 0.6596589189007366,
 0.7834269568434862,
 0.4950553331763598,
 0.6798664603585725,
 0.7433566551178983,
 0.6989976117595613,
 0.7803659726193279,
 0.7109892697366208,
 0.5466657694636969,
 0.7256072942601439]
'''
cv_score = []

for train_test_index in splitter.split(labels, labels):
    train_labels = labels.reset_index('cutoff', drop=True).iloc[train_test_index[0]]
    test_labels = labels.reset_index('cutoff', drop=True).iloc[train_test_index[1]]
    train_fm = fm.reset_index('time', drop=True).loc[train_labels.index, :]
    test_fm = fm.reset_index('time', drop=True).loc[test_labels.index, :]

    dl_model.fit(
        train_fm, train_labels, fl=fl,
        batch_size=128,
        workers=8,
        use_multiprocessing=True,
        shuffle=False,
        epochs=3)
    break

    predictions = dl_model.predict(test_fm)
    score = roc_auc_score(test_labels, predictions)
    print("cv score: ", score)
    cv_score.append(score)
mean_score = np.mean(cv_score)
stderr = 2 * (np.std(cv_score) / np.sqrt(n_splits))

print("DFS AUC %.2f +/- %.2f" % (mean_score, stderr))

# categorical_feature_names = ["model"]
# dl_model.compile(cutoff_raw, categorical_feature_names=categorical_feature_names)

# cv_score = []

# for i, train_test_index in enumerate(splitter.split(labels, labels)):
    # train_labels = labels.reset_index('cutoff', drop=True).iloc[train_test_index[0]]
    # test_labels = labels.reset_index('cutoff', drop=True).iloc[train_test_index[1]]
    # train_fm = cutoff_raw.reset_index('date', drop=True).loc[train_labels.index, :]
    # test_fm = cutoff_raw.reset_index('date', drop=True).loc[test_labels.index, :]

    # dl_model.fit(
        # train_fm, train_labels,
        # batch_size=128,
        # epochs=3,
        # validation_split=0.1)

    # predictions = dl_model.predict(test_fm)
    # score = roc_auc_score(test_labels, predictions)
    # print("cv score: ", score)
    # cv_score.append(score)

# mean_score = np.mean(cv_score)
# stderr = 2 * (np.std(cv_score) / np.sqrt(n_splits))

# print("DENORM AUC %.2f +/- %.2f" % (mean_score, stderr))
