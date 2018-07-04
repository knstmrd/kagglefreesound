from typing import Dict, List


def write_list(fname: str, list_to_write: List[str], mode: str='a', ending=None):
    with open(fname, mode) as f:
        for item in list_to_write:
            f.write(item)
            f.write('\n')
        if ending is not None:
            f.write(ending)


def log_cv_results(classifier, score: float, n_features, settings: Dict):
    log = ['Score: ' + str(score)]
    log += ['Classifier: ' + str(classifier)]
    log += ['Feature file: ' + settings['features_file']]
    log += ['Correlated features files: ' + '_'.join(settings['correlated_features_files'])]
    log += ['Unimportant features files: ' + '_'.join(settings['unimportant_features_files'])]
    log += ['Total # of used features ' + str(n_features)]

    write_list('logs/cv/' + settings['method'] + '_cv.log', log, 'a', '\n\n')


def log_submission(classifier, n_features, settings: Dict):
    log = ['Features files: ' + ','.join(settings['features_files'])]
    log.append('Exclusion files: ' + ','.join(settings['exclude_features_files']))
    log.append('Total # of used features ' + str(n_features))
    log.append(str(classifier))

    if 'prob_threshold' in settings:
        log.append('Probability threshold: ' + str(settings['prob_threshold']))
    write_list('logs/submissions/' + settings['method'] + '/submissions.log', log, 'a', '\n\n')
