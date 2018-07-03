import pandas as pd
import numpy as np
from writelogs import write_list
from typing import Dict, List


def load_and_merge_dataframes(path: str, keys: List[str], merge_on='fname',
                              exclude_cols_all=None,
                              exclude_cols_except_first=None):
    """
    Load dataframes from a HDF5 table, merge into one, while dropping some columns
    :param path:
    :param keys:
    :param merge_on:
    :param exclude_cols_all:
    :param exclude_cols_except_first:
    :return:
    """
    if exclude_cols_except_first is None:
        exclude_cols_except_first = []
    if exclude_cols_all is None:
        exclude_cols_all = []

    df_list = [pd.read_hdf(path, key=key) for key in keys]

    if len(df_list) == 1:
        base_df = df_list[0]
    else:
        base_df = df_list[0]
        for df in df_list[1:]:
            if exclude_cols_except_first is not None:
                use_cols = [col for col in df.columns if col not in exclude_cols_except_first]
            else:
                use_cols = df.columns
            base_df = base_df.merge(df[use_cols], on=merge_on)

    if exclude_cols_all is not None:
        remove_cols = [col for col in base_df.columns if col in exclude_cols_all]
        base_df.drop(remove_cols, axis=1, inplace=True)

    return base_df


def find_correlated_features(df: pd.DataFrame, fname: str, settings: Dict, exclude_cols=None):
    """
    Find and write a list of features that are highly correlated
    :param df:
    :param fname:
    :param settings:
    :param exclude_cols:
    :return:
    """
    if exclude_cols is None:
        feat_cols = df.columns
    else:
        feat_cols = [col for col in df.columns if col not in exclude_cols]
    corr = df[feat_cols].corr()
    corr = corr.abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > settings['correlation_threshold'])]
    print(len(to_drop), 'higly correlated features found, threshold =', str(settings['correlation_threshold']))
    write_list('correlated_features/' + fname + '_threshold_' + str(settings['correlation_threshold']),
               to_drop, 'a')

    print('Wrote into ' + 'correlated_features/' + fname + '_threshold_' + str(settings['correlation_threshold']))
    return fname + '_threshold_' + str(settings['correlation_threshold']), to_drop


def find_unimportant_features(fname: str, feat_cols, feature_importances, settings: Dict):
    """
    Find and write a list of features that have a low importance (in tree-based models)
    :param fname:
    :param feat_cols:
    :param feature_importances:
    :param settings:
    :return:
    """
    importances = list(zip(feat_cols, feature_importances))
    importances.sort(key=lambda x: x[1])

    importances_vals = np.array([x[1] for x in importances])
    cutoff = np.sum(importances_vals < settings['importance_threshold'])
    unimportant_feature_names = [fn[0] for fn in importances[:cutoff]]
    print(cutoff, ' unimportant features found')
    write_list('unimportant_features/' + fname + '_threshold_' + str(settings['importance_threshold']),
               unimportant_feature_names, 'a')
    print('Wrote into ' + 'unimportant_features/' + fname + '_threshold_' + str(settings['importance_threshold']))
    return fname + '_threshold_' + str(settings['importance_threshold'])


def cleanup_features(df: pd.DataFrame, fname: str, settings, exclude_cols=None):
    """
    Given a list of filenames containing lists of highly correlated and unimportant features,
    and a dataframe, write a list of remaining features and write some data to identify the feature set
    (so that if new columns are added, the code re-computes the correlation and importances parts
    for the new columns, while the thrown-out columns remain thrown out)
    We return the cleaned-up feature list
    :param df:
    :param fname:
    :param settings:
    :param exclude_cols:
    :return:
    """
    if exclude_cols is None:
        feat_cols = df.columns
    else:
        feat_cols = [col for col in df.columns if col not in exclude_cols]

    remove_features = []
    log_list = []

    if 'correlated_features_files' in settings:
        for corr_fname in settings['correlated_features_files']:
            with open('correlated_features/' + corr_fname) as f:
                correlated_feat_cols = [line.replace('\n', '') for line in f]
            remove_features += correlated_feat_cols

        log_list.append('Correlated feature files: ' + ' '.join(settings['correlated_features_files']))
    else:
        log_list.append('Correlated feature files: ')

    if 'unimportant_features_files' in settings:
        for corr_fname in settings['unimportant_features_files']:
            with open('unimportant_features/' + corr_fname) as f:
                unimportant_feat_cols = [line.replace('\n', '') for line in f]
            remove_features += unimportant_feat_cols
        log_list.append('Unimportant feature files: ' + ' '.join(settings['unimportant_features_files']))
    else:
        log_list.append('Unimportant feature files: ')

    remove_features = set(remove_features)
    print('Total features to remove: ' + str(len(remove_features)))

    feat_cols_cleaned = [col for col in feat_cols if col not in remove_features]
    print(str(len(feat_cols_cleaned)) + ' features after removal')

    write_list('feature_lists/' + fname, feat_cols_cleaned, 'w')
    print('Wrote feature list into feature_lists/' + fname)

    log_list.append('Features file: ' + 'feature_lists/' + fname)
    log_list.append('Total # of columns in dataframe: ' + str(len(df.columns)))
    log_list.append('Features used: ' + str(len(feat_cols_cleaned)))
    write_list('logs/data.log', log_list, 'w')
    return feat_cols_cleaned


def load_settings():
    settings = {}
    with open('logs/data.log', 'r') as f:
        raw_data = [line.replace('\n', '') for line in f]
    for line in raw_data:
        split_line = line.split(': ')
        if split_line[0].startswith('Correlated feature files'):
            settings['correlated_features_files'] = split_line[1].split(' ')
        elif split_line[0].startswith('Unimportant feature files'):
            settings['unimportant_features_files'] = split_line[1].split(' ')
        elif split_line[0].startswith('Features file'):
            settings['features_file'] = split_line[1]
        elif split_line[0].startswith('Features used'):
            settings['features_used'] = int(split_line[1])
        else:
            settings['total_number_of_columns'] = int(split_line[1])
    return settings


def process_df(df, fname, settings, exclude_cols=None):
    """
    Caveats: if feature set is the same, but change correlation or importance thresholds are changed,
    find_correlated_features, find_unimportant_features have to be re-run manually
    :param df:
    :param fname:
    :param settings:
    :param exclude_cols:
    :return:
    """
    rerun_correlation = False
    try:
        feature_settings = load_settings()
        if feature_settings['total_number_of_columns'] != len(df.columns):
            rerun_correlation = True
    except FileNotFoundError as e:
        feature_settings = {'correlated_features_files': [],
                            'unimportant_features_files': [],
                            'features_file': []}
        rerun_correlation = True
    merged_settings = {**settings, **feature_settings}
    remove_features = []
    # TODO: fix and feature_settings['unimportant_features_files'][0] != ''

    if rerun_correlation:
        print('Re-computing correlated features')
        # perhaps we already have some pre-computed lists of features we're not using,
        # so let's remove them first
        for corr_fname in feature_settings['correlated_features_files']:
            with open('correlated_features/' + corr_fname) as f:
                feat_cols = [line.replace('\n', '') for line in f]
            remove_features += feat_cols

        if feature_settings['unimportant_features_files'] != []:
            for unimportant_fname in feature_settings['unimportant_features_files']:
                with open('unimportant_features/' + unimportant_fname) as f:
                    feat_cols = [line.replace('\n', '') for line in f]
                remove_features += feat_cols

        corr_fname, corr_features = find_correlated_features(df, fname, merged_settings, exclude_cols+remove_features)
        merged_settings['correlated_features_files'] += [corr_fname]
        feat_cols_cleaned = cleanup_features(df, fname, merged_settings, exclude_cols)
    else:  # don't need to recompute correlations, and we can load feature names from a file
        print('Nothing to re-compute, loading feature names from file')
        with open(merged_settings['features_file']) as f:
            feat_cols_cleaned = [line.replace('\n', '') for line in f]
    return merged_settings, feat_cols_cleaned


def run_cv():
    pass


def run_classification():
    pass
