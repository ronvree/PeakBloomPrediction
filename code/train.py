

import pandas as pd
import numpy as np

import torch
import torch.optim
import torch.utils.data

import code.data as data
import code.model as model


def batch_tensors(*ts):
    return torch.cat([t.unsqueeze(0) for t in ts], dim=0)


def _collate_fn(samples: list):

    years = [sample['year'] for sample in samples]
    locations = [sample['location'] for sample in samples]
    dates = [sample['dates'] for sample in samples]

    # print([sample['bloom_doy'] for sample in samples])

    temp_max = batch_tensors(*[sample['temp_max'] for sample in samples])
    temp_min = batch_tensors(*[sample['temp_min'] for sample in samples])
    temp_mean = batch_tensors(*[sample['temp_mean'] for sample in samples])
    bloom_doy = batch_tensors(*[sample['bloom_doy'] for sample in samples])
    lat = batch_tensors(*[sample['lat'] for sample in samples])
    lon = batch_tensors(*[sample['lon'] for sample in samples])
    alt = batch_tensors(*[sample['alt'] for sample in samples])

    return {
        'year': years,
        'location': locations,
        'dates': dates,
        'temp_max': temp_max,
        'temp_min': temp_min,
        'temp_mean': temp_mean,
        'bloom_doy': bloom_doy,
        'lat': lat,
        'lon': lon,
        'alt': alt,
    }


def train_model(m=None):

    m = m or model.PeakBlossomPredictionModel()

    dataset = data.TemperatureDataset()

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=16,
                                             shuffle=True,
                                             collate_fn=_collate_fn
                                             )

    optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

    loss_f = torch.nn.MSELoss()

    for _ in range(20):
        for sample in dataloader:

            t_max = sample['temp_max']
            t_min = sample['temp_min']
            t_mean = sample['temp_mean']

            lat = sample['lat']
            lon = sample['lon']
            alt = sample['alt']

            bloom_doy = sample['bloom_doy']

            optimizer.zero_grad()

            bloom_doy_pred = m(t_max, t_min, t_mean, lat, lon, alt)

            loss = loss_f(bloom_doy_pred * 365, bloom_doy)
            print(loss)

            loss.backward()

            optimizer.step()



if __name__ == '__main__':

    train_model()
