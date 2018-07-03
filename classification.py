import pandas as pd
import numpy as np
import feature_processing as fp
from kaggle_stuff import *  # file with all the classifier settings, and misc stuff

fname = 'testingrightnow'
df_train = fp.load_and_merge_dataframes('data/features.h5', ['train_basic', 'train_v2'],
                                        exclude_cols_all=['manually_verified', 'stft length'],
                                        exclude_cols_except_first=['label', 'manually_verified'])

labels = df_train['label']

settings, features_cleaned = fp.process_df(df_train, fname, settings, exclude_cols=['label', 'fname'])
print(len(features_cleaned))
