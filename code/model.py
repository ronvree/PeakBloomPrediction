import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM


class TemperatureModel(nn.Module):

    def __init__(self, hidden_size=128):
        super().__init__()

        self._input_size = 3
        self._hidden_size = hidden_size

        self._rnn = LSTM(input_size=self._input_size,
                         hidden_size=self._hidden_size,
                         batch_first=True,
                         )

    def forward(self, t_max, t_min, t_mean):

        xs = torch.cat([t_max.unsqueeze(-1),
                        t_min.unsqueeze(-1),
                        t_mean.unsqueeze(-1)],
                       dim=2)

        h0 = torch.randn(1, xs.shape[0], self._hidden_size)
        c0 = torch.randn(1, xs.shape[0], self._hidden_size)

        xss, _ = self._rnn(xs, (h0, c0))  # shape: (N, L, H)
        xs = xss[:, -1, :].squeeze()

        return xs


class PeakBlossomPredictionModel(nn.Module):

    def __init__(self):
        super().__init__()

        input_size = 3
        hidden_size = 128

        self._temp_model = TemperatureModel(hidden_size=hidden_size)
        self._dense_1 = nn.Linear(3 + hidden_size, 128)
        self._dense_2 = nn.Linear(128, 1)

    def forward(self, t_max, t_min, t_mean, lat, lon, alt):

        xs = self._temp_model(t_max, t_min, t_mean)
        xs = F.relu(xs)

        xs = torch.cat([xs, lat, lon, alt], dim=1)

        xs = self._dense_1(xs)
        xs = F.relu(xs)

        xs = self._dense_2(xs)

        return xs


if __name__ == '__main__':

    model = TemperatureModel()

    tmax = torch.randn(5, 10)
    tmin = torch.randn(5, 10)
    tmean = torch.randn(5, 10)

    xs = model(tmax, tmin, tmean)

    print(xs.shape)
