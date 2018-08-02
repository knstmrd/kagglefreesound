import pandas as pd
import numpy as np
from datetime import datetime
from kaggle_stuff import *

fname = str(datetime.now()).replace(':', '_').replace(' ', '_')[5:19]
prob_arrs_names = ['probs_lgbm/07-31_15_27_23.npy', 'probs_cb/07-31_14_16_39.npy']  # 0.926, 0.889

prob_arrs = [np.load('submissions/' + x) for x in prob_arrs_names]

use_geometric = False
use_geometric = True

weights_mean = [0.75, 0.25]
weights_geom = [0.75, 0.25]

filenames = pd.read_hdf('data/features_new.h5', key='test_v1_v2', usecols=['fname'])

if use_geometric:
    output_prob = np.ones_like(prob_arrs[0])
    for w, arr in zip(weights_geom, prob_arrs):
        output_prob *= arr**w

else:
    output_prob = np.zeros_like(prob_arrs[0])
    for w, arr in zip(weights_mean, prob_arrs):
        output_prob += arr * w

print(output_prob.shape)

output_labels = []

for i in range(output_prob.shape[0]):
    top_3 = output_prob[i, :].argsort()[-3:][::-1]
    if output_prob[i, top_3[0]] < multiple_labels_threshold:
        output_labels.append([labels_list[x] for x in top_3])
    else:
        output_labels.append([labels_list[top_3[0]]])

output_labels = [' '.join(ll) for ll in output_labels]
output_df = {'fname': filenames['fname'],
             'label': output_labels}
output_df = pd.DataFrame(output_df)

output_df.to_csv('submissions/avg/' + fname
                 + '_thresh' + str(multiple_labels_threshold) + '.csv', index=False, header=True)

weights_geom_str = [str(x) for x in weights_geom]
weights_mean_str = [str(x) for x in weights_mean]

with open('logs/submissions/avg.log', 'a') as f:
    f.write(fname + '\n')
    f.write(', '.join(prob_arrs_names) + '\n')
    if use_geometric:
        f.write('Geometric mean, weights: ' + ', '.join(weights_geom_str))
    else:
        f.write('Arithmetic mean, weights: ' + ', '.join(weights_mean_str))
    f.write('\n\n\n')
