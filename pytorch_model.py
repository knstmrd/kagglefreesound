import torch.nn as nn
from kaggle_stuff import *
from math import floor


def l_out(l_in, dilation, kernel_size, stride, padding=0):
    return floor((l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


class RawWave(nn.Module):
    def __init__(self, dilation1=1, dilation2=1):
        super(RawWave, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, dilation=dilation1),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=9, dilation=dilation2),
            nn.ReLU(),
            nn.MaxPool1d(16),
            nn.BatchNorm1d(16),
            nn.Dropout(p=0.1)
        )
        self.l1_out = l_out(l_out(l_out(audio_input_length, dilation1, 9, 1), dilation2, 9, 1), 1, 16, 16)

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, dilation=dilation1),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, dilation=dilation2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.1)
        )
        self.l2_out = l_out(l_out(l_out(self.l1_out, dilation1, 3, 1), dilation2, 3, 1), 1, 4, 4)

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, dilation=dilation1),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, dilation=dilation2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.1)
        )
        self.l3_out = l_out(l_out(l_out(self.l2_out, dilation1, 3, 1), dilation2, 3, 1), 1, 4, 4)

        self.l4_out = l_out(l_out(self.l3_out, dilation1, 3, 1), dilation2, 3, 1)

        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=256, kernel_size=3, dilation=dilation1),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, dilation=dilation2),
            nn.ReLU(),
            nn.MaxPool1d(self.l4_out),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.23)
        )

        self.layer5 = nn.Sequential(
            nn.Linear(in_features=256, out_features=1024),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=64),
            nn.ReLU()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=64, out_features=pytorch_settings['num_classes']),
            nn.Softmax()
        )

    def forward_embed(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out[:, :, 0]
        out = self.layer5(out)
        out = self.layer6(out)
        return out

    def forward(self, x):
        return self.output_layer(self.forward_embed(x))
