import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from feature_processing import mapk


LABELS = ["Acoustic_guitar", "Applause", "Bark", "Bass_drum",
          "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet",
          "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close",
          "Electric_piano", "Fart", "Finger_snapping", "Fireworks", "Flute",
          "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica",
          "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow",
          "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter",
          "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet",
          "Violin_or_fiddle", "Writing"]

train = pd.read_csv("train.csv", usecols=['fname', 'label'])
# test = pd.read_csv("test.csv")

train.set_index("fname", inplace=True)
# test.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: LABELS.index(x))
labels = train["label_idx"]

skf = StratifiedKFold(n_splits=5, random_state=5)
truth = []

for train_index, test_index in skf.split(labels, labels):
    truth.append(labels[test_index])

cv_cb = [np.load('logs/cv/probs_cb/07-31_09_54_52___%d.npy' % i) for i in range(1, 6)]
cv_lgbm = [np.load('logs/cv/probs_lgbm/07-30_22_12_15___%d.npy' % i) for i in range(1, 6)]

w_ar = np.linspace(0., 1.0, num=25)

result = []
for w in w_ar:
    # curr_cv
    fold_counter = 0
    fold_scores = []
    for i in range(5):
        curr_truth = [[v] for v in truth[i]]
        # truth += curr_truth
        curr_predictions = []
        predictions_proba = cv_cb[i] * (1 - w) + cv_lgbm[i] * w
        for i in range(predictions_proba.shape[0]):
            top_3 = predictions_proba[i, :].argsort()[-3:][::-1]
            curr_predictions.append([x for x in top_3])

        fold_scores.append(mapk(curr_truth, curr_predictions, 3))
    print(w, 'Mean scores: ' + str(np.mean(fold_scores)) + ' std scores: ' + str(np.std(fold_scores)))
    result.append([w, str(np.mean(fold_scores))])

w_g = np.linspace(0., 1.0, num=25)

result_g = []
for w in w_g:
    # result = []
    # curr_cv
    fold_counter = 0
    fold_scores = []
    for i in range(5):
        curr_truth = [[v] for v in truth[i]]
        # truth += curr_truth
        curr_predictions = []
        predictions_proba = (cv_cb[i] ** (1 - w)) * (cv_lgbm[i] ** w)
        for i in range(predictions_proba.shape[0]):
            top_3 = predictions_proba[i, :].argsort()[-3:][::-1]
            curr_predictions.append([x for x in top_3])

        fold_scores.append(mapk(curr_truth, curr_predictions, 3))
    print(w, 'Mean scores: ' + str(np.mean(fold_scores)) + ' std scores: ' + str(np.std(fold_scores)))
    result_g.append([w, str(np.mean(fold_scores))])

result.sort(key=lambda x: x[1], reverse=True)
print('#' * 20)
print(result[:10])
print('#' * 20)

result_g.sort(key=lambda x: x[1], reverse=True)
print('#' * 20)
print(result_g[:10])
print('#' * 20)
