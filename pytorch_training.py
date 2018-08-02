import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from kaggle_stuff import *
from pytorch_model import RawWave
from sklearn.model_selection import StratifiedKFold
from pytorch_dataloader import AudioDataset
from torch.utils.data import DataLoader
from feature_processing import mapk

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
df_train = pd.read_hdf('data/features_new.h5', key='train_v1_v2', usecols=['fname', 'label'])
df_test = pd.read_hdf('data/features_new.h5', key='test_v1_v2', usecols=['fname'])

model = RawWave().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=pytorch_settings['learning_rate'])
bs = pytorch_settings['batch_size']

labels = df_train['label']
fnames_train = df_train['fname']

skf = StratifiedKFold(n_splits=settings['N_folds'], random_state=5)

for fold, (train_index, test_index) in enumerate(skf.split(labels, labels)):
    print('Fold {}/{}'.format(fold+1, settings['N_folds']))
    curr_predictions = []
    curr_truth = [[v] for v in labels[test_index].values]

    train_dataset = AudioDataset(path_to_audio, fnames_train[train_index].values, labels[train_index].values,
                                 pytorch_settings['sample_rate'])
    train_dataloader = DataLoader(train_dataset, batch_size=pytorch_settings['batch_size'],
                                  shuffle=True, num_workers=2)

    test_dataset = AudioDataset(path_to_audio, fnames_train[test_index].values, labels[test_index].values,
                                pytorch_settings['sample_rate'])
    test_dataloader = DataLoader(test_dataset, batch_size=pytorch_settings['batch_size'],
                                 shuffle=True, num_workers=2)
    best_result = 0.0
    for epoch in range(pytorch_settings['max_epochs']):
        model.train()
        for i_batch, sample_batched in enumerate(train_dataloader):
            outputs = model(sample_batched['audio'].resize_(sample_batched['audio'].shape[0], 1, audio_input_length))
            loss = criterion(outputs, sample_batched['labels'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if i_batch > 5:
            #     break
            if i_batch % 5 == 0:
                print('Epoch [{}/{}], [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, pytorch_settings['max_epochs'], i_batch, len(train_dataloader), loss.item()))
        model.eval()
        with torch.no_grad():
            print('Predicting')
            predictions_proba = np.zeros((len(test_index), 41))
            offset = 0
            for i_batch, sample_batched in enumerate(test_dataloader):
                print('[{}/{}]'.format(i_batch, len(test_dataloader)))
                outputs = model(sample_batched['audio'].resize_(sample_batched['audio'].shape[0], 1, audio_input_length))
                np_outputs = outputs.numpy()
                predictions_proba[offset:offset + np_outputs.shape[0], :] = np_outputs
                offset += np_outputs.shape[0]

            for i in range(predictions_proba.shape[0]):
                top_3 = predictions_proba[i, :].argsort()[-3:][::-1]
                if predictions_proba[i, top_3[0]] < multiple_labels_threshold:
                    curr_predictions.append([x for x in top_3])
                else:
                    curr_predictions.append([labels_list[top_3[0]]])
            print(mapk(curr_truth, curr_predictions, 3))
            # TODO: early stopping
            # TODO: write embeddings (train OOF)
            # TODO: write embeddings (test)
