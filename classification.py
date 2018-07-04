import pandas as pd
import numpy as np
import feature_processing as fp
from datetime import datetime
from tqdm import tqdm
from kaggle_stuff import *  # file with all the classifier settings, and misc stuff
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from writelogs import log_cv_results, log_submission

fname = str(datetime.now()).replace(':', '_').replace(' ', '_')[5:19]
df_train = fp.load_and_merge_dataframes('data/features.h5', ['train_basic', 'train_v2', 'train_v3', 'train_v4'],
                                        exclude_cols_all=['manually_verified', 'stft length'],
                                        exclude_cols_except_first=['label', 'manually_verified'])

df_test = fp.load_and_merge_dataframes('data/features.h5', ['test_basic', 'test_v2', 'test_v3', 'test_v4'],
                                       exclude_cols_all=['manually_verified', 'stft length'],
                                       exclude_cols_except_first=['label', 'manually_verified'])

labels = df_train['label']

settings, feat_cols_cleaned = fp.process_df(df_train, fname, settings, exclude_cols=['label', 'fname'],
                                            force_rerun_correlation=False)
print(len(feat_cols_cleaned))

ss = StandardScaler()
ss.fit(df_train[feat_cols_cleaned].values)
train_scaled = ss.transform(df_train[feat_cols_cleaned].values)

if 'do_cv' in settings.keys():
    scores = []
    importances = []

    pg = ParameterGrid(param_grid[settings['method']])

    for param_set in tqdm(pg):
        if settings['method'] == 'lgbm':
            classifier = lgb.LGBMClassifier(**param_set, objective='multiclass')
        else:
            classifier = LogisticRegression(C=1)

        skf = StratifiedKFold(n_splits=settings['N_folds'], random_state=5)
        truth = []
        predictions = []

        for train_index, test_index in skf.split(train_scaled, labels):
            print('Fold number', str(len(truth)+1))
            classifier.fit(train_scaled[train_index, :], labels[train_index])
            truth.append(labels[test_index].values)
            predictions.append(classifier.predict(train_scaled[test_index, :]))

        score = precision_score(np.hstack(truth), np.hstack(predictions), average='macro')
        scores.append(score)

        log_cv_results(classifier, score, len(feat_cols_cleaned), settings)

    scores_params = list(zip(list(pg), scores))
    scores_params.sort(key=lambda x: x[1], reverse=True)
    print('Best settings: ' + str(scores_params[0]))
    print('Worst settings: ' + str(scores_params[-1]))
else:
    if settings['method'] == 'lgbm':
        classifier = lgb.LGBMClassifier(**lgbm_settings, objective='multiclass')
    else:
        classifier = LogisticRegression(C=1)
    classifier.fit(train_scaled, labels)

    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test.fillna(0, inplace=True)
    test_scaled = ss.transform(df_test[feat_cols_cleaned])

    if settings['method'] == 'lgbm' and settings['importance_threshold'] >= 0:
        if settings['unimportant_features_files'] == []:
            settings['unimportant_features_files'] = [fp.find_unimportant_features(fname, feat_cols_cleaned,
                                                                                   classifier.feature_importances_,
                                                                                   settings)]
        else:
            settings['unimportant_features_files'] += fp.find_unimportant_features(fname, feat_cols_cleaned,
                                                                                   classifier.feature_importances_,
                                                                                   settings)
        fp.cleanup_features(df_train, fname, settings, exclude_cols=['label', 'fname'])
        # TODO: log all feature importances!, add pre-processing step so that we can filter when we load

    probs = classifier.predict_proba(test_scaled)
    output_labels = []
    uncertain_predictions = 0

    for i in range(probs.shape[0]):
        top_3 = probs[i, :].argsort()[-3:][::-1]
        if probs[i, top_3[0]] < multiple_labels_threshold:
            output_labels.append([labels_list[x] for x in top_3])
            uncertain_predictions += 1
        else:
            output_labels.append([labels_list[top_3[0]]])

    output_labels = [' '.join(ll) for ll in output_labels]
    output_df = {'fname': df_test['fname'],
                 'label': output_labels}
    output_df = pd.DataFrame(output_df)

    output_df.to_csv('submissions/' + settings['method'] + '/' + fname
                     + '_thresh' + str(multiple_labels_threshold) + '.csv', index=False, header=True)

    with open('logs/submissions/' + settings['method'] + '_submissions.log', 'a') as f:
        f.write('submission filename: ' + fname + '\n')
        f.write(str(classifier))
        f.write('\nThreshold: ' + str(multiple_labels_threshold))
        if 'features_file' in settings:
            if type(settings['features_file']) is list:
                f.write('\nfeatures: ' + settings['features_file'][0] + '\n')
            else:
                f.write('\nfeatures: ' + settings['features_file'] + '\n')
        elif 'features_file_v2' in settings:
            f.write('\nfeatures: ' + settings['features_file_v2'] + '\n')
        else:
            f.write('\n'.join(feat_cols_cleaned))
        f.write('\n\n')
