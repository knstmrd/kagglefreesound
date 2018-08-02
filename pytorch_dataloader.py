import numpy as np
from kaggle_stuff import *
from torch.utils.data import Dataset
from math import floor
import librosa as lr

input_length = floor(pytorch_settings['sample_seconds'] * pytorch_settings['sample_rate'])


def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5


def get_batch(fnames, labels=None):
    output = np.zeros((pytorch_settings['batch_size'], 1, input_length))

    if labels is not None:
        addpth = 'audio_train/'
        Y = labels[:pytorch_settings['batch_size']]
    else:
        addpth = 'audio_test/'
        Y = None
    for i, fname in enumerate(fnames):
        if pytorch_settings['resample']:
            data, _ = lr.core.load(path_to_audio + addpth + fname, sr=pytorch_settings['sample_rate'],
                                   res_type='kaiser_fast')
        else:
            data, sr = lr.core.load(path_to_audio + addpth + fname, sr=None)
            assert(sr == pytorch_settings['sample_rate'])

        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length + offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), 'wrap')
        data = audio_norm(data)
        output[i, 0, :] = data
    return output, Y


class AudioDataset(Dataset):
    def __init__(self, base_path, fnames, labels=None, sr=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.fnames = fnames
        self.labels = labels
        self.sr = sr

        self.base_path = base_path

        if labels is None:
            self.base_path += '/audio_test/'
        else:
            self.base_path += '/audio_train/'
        # self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        data, _ = lr.core.load(self.base_path + self.fnames[idx], sr=self.sr)

        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length + offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), 'wrap')
        data = audio_norm(data)

        sample = {'audio': data, 'fname': self.fnames[idx]}
        if self.labels is None:
            sample['labels'] = None
        else:
            sample['labels'] = self.labels[idx]

        return sample
