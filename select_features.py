from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier,
                              RandomForestRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.base import (ClassifierMixin, RegressorMixin)
from sklearn.preprocessing import StandardScaler, Imputer, FunctionTransformer
from sklearn.metrics import (f1_score,
                             mean_absolute_error,
                             roc_auc_score,
                             r2_score)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import utils_backblaze as utils
import pandas as pd
import featuretools as ft

data_dir = '../backblaze_harddrive_data'
df = utils.load_data_as_dataframe(data_dir=data_dir, csv_glob='*.csv', nrows=10)
es = utils.load_entityset_from_dataframe(df)
print("loaded es")
fm = pd.read_csv('backblaze_ftens_high_info.csv', parse_dates=['time'], index_col=['serial_number', 'time']).sort_index()
fm.to_csv("backblaze_ftens_sorted.csv")
labels = pd.read_csv('backblaze_labels.csv', parse_dates=['cutoff'], index_col=['serial_number', 'cutoff'])['label'].sort_index()
print("loaded labels")
fl = ft.load_features('backblaze_high_info_fl.p', es)
fm, fl = ft.encode_features(fm, fl)
print("encoded")
fm = fm.groupby(level='serial_number').last()
print("grouped")
fm.to_csv("backblaze_ftens_sorted_last.csv")
labels.to_frame().to_csv("backblaze_labels_sorted.csv")

# fm = pd.read_csv("backblaze_ftens_sorted_last.csv",parse_dates=['time'], index_col=['serial_number', 'time'])
# labels = pd.read_csv('backblaze_labels_sorted.csv', parse_dates=['cutoff'], index_col=['serial_number', 'cutoff'])['label']
# est = RandomForestClassifier(n_estimators=1000, class_weight='balanced', n_jobs=-1, verbose=True)
# imputer = Imputer(missing_values='NaN', strategy="mean", axis=0)
# selector = SelectFromModel(est, threshold="mean")
# pipeline = Pipeline([("imputer", imputer),
                         # ("scaler", StandardScaler()),
                         # ("selector", selector)])
# pipeline.fit(fm, labels)
# selected = set(fm.loc[:, pipeline.steps[-1][1].get_support()].columns.tolist())
# fl_selected = [f for f in fl if f.get_name() in selected]
# ft.save_features(fl_selected, "fl_backblaze_selected.p")
