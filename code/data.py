import numpy as np
import pandas as pd

import torch
import torch.utils.data

"""

    Original Competition Data

"""

DATA_PATH = '../data'
DATA_PATH_JAPAN = f'{DATA_PATH}/japan.csv'
DATA_PATH_KYOTO = f'{DATA_PATH}/kyoto.csv'
DATA_PATH_LIESTAL = f'{DATA_PATH}/liestal.csv'
DATA_PATH_METEOSWISS = f'{DATA_PATH}/meteoswiss.csv'
DATA_PATH_SOUTH_KOREA = f'{DATA_PATH}/south_korea.csv'
DATA_PATH_WASHINGTONDC = f'{DATA_PATH}/washingtondc.csv'

DATA_PATH_INDIVIDUAL_PHENOMETRICS_DATA = f'{DATA_PATH}/USA-NPN_individual_phenometrics_data.csv'
DATA_PATH_INDIVIDUAL_PHENOMETRICS_DATAFIELD_DESCRIPTIONS = f'{DATA_PATH}/USA-NPN_individual_phenometrics_datafield_descriptions.csv'
DATA_PATH_INTENSITY_OBSERVATIONS_DATA = f'{DATA_PATH}/USA-NPN_status_intensity_observations_data.csv'
DATA_PATH_INTENSITY_DATAFIELD_DESCRIPTIONS = f'{DATA_PATH}/USA-NPN_status_intensity_datafield_descriptions.csv'


def get_data_japan() -> pd.DataFrame:
    """

    ['location', 'lat', 'long', 'alt', 'year', 'bloom_date', 'bloom_doy']

    :return:
    """
    df = pd.read_csv(DATA_PATH_JAPAN)
    return df


def get_data_kyoto() -> pd.DataFrame:
    """

    ['location', 'lat', 'long', 'alt', 'year', 'bloom_date', 'bloom_doy']

    :return:
    """
    df = pd.read_csv(DATA_PATH_KYOTO)
    return df


def get_data_liestal() -> pd.DataFrame:
    """

    ['location', 'lat', 'long', 'alt', 'year', 'bloom_date', 'bloom_doy']

    :return:
    """
    df = pd.read_csv(DATA_PATH_LIESTAL)
    return df


def get_data_meteoswiss() -> pd.DataFrame:
    """

    ['location', 'lat', 'long', 'alt', 'year', 'bloom_date', 'bloom_doy']

    :return:
    """
    df = pd.read_csv(DATA_PATH_METEOSWISS)
    return df


def get_data_south_korea() -> pd.DataFrame:
    """

    ['location', 'lat', 'long', 'alt', 'year', 'bloom_date', 'bloom_doy']

    :return:
    """
    df = pd.read_csv(DATA_PATH_SOUTH_KOREA)
    return df


def get_data_washingtondc() -> pd.DataFrame:
    """

    ['location', 'lat', 'long', 'alt', 'year', 'bloom_date', 'bloom_doy']

    :return:
    """
    df = pd.read_csv(DATA_PATH_WASHINGTONDC)
    return df


def get_individual_phenometrics_data() -> pd.DataFrame:
    """

    ['Site_ID', 'Latitude', 'Longitude', 'Elevation_in_Meters', 'State',
       'Species_ID', 'Genus', 'Species', 'Common_Name', 'Kingdom',
       'Individual_ID', 'Phenophase_ID', 'Phenophase_Description',
       'First_Yes_Year', 'First_Yes_Month', 'First_Yes_Day', 'First_Yes_DOY',
       'First_Yes_Julian_Date', 'NumDays_Since_Prior_No', 'Last_Yes_Year',
       'Last_Yes_Month', 'Last_Yes_Day', 'Last_Yes_DOY',
       'Last_Yes_Julian_Date', 'NumDays_Until_Next_No', 'AGDD', 'AGDD_in_F',
       'Tmax_Winter', 'Tmax_Spring', 'Tmin_Winter', 'Tmin_Spring',
       'Prcp_Winter', 'Prcp_Spring']

    :return:
    """
    df = pd.read_csv(DATA_PATH_INDIVIDUAL_PHENOMETRICS_DATA)
    return df


def get_individual_phenometrics_datafield_descriptions() -> pd.DataFrame:
    """

    ['Field name', 'Field description', 'Controlled value choices']

    :return:
    """
    df = pd.read_csv(DATA_PATH_INDIVIDUAL_PHENOMETRICS_DATAFIELD_DESCRIPTIONS)
    return df


def get_status_intensity_observations_data() -> pd.DataFrame:
    """

    ['Observation_ID', 'Update_Datetime', 'Site_ID', 'Latitude', 'Longitude',
       'Elevation_in_Meters', 'State', 'Species_ID', 'Genus', 'Species',
       'Common_Name', 'Kingdom', 'Individual_ID', 'Phenophase_ID',
       'Phenophase_Description', 'Observation_Date', 'Day_of_Year',
       'Phenophase_Status', 'Intensity_Category_ID', 'Intensity_Value',
       'Abundance_Value', 'AGDD', 'AGDD_in_F', 'Tmax_Winter', 'Tmax_Spring',
       'Tmin_Winter', 'Tmin_Spring', 'Prcp_Winter', 'Prcp_Spring',
       'Daylength']

    :return:
    """
    df = pd.read_csv(DATA_PATH_INTENSITY_OBSERVATIONS_DATA)
    return df


def get_status_intensity_datafield_descriptions() -> pd.DataFrame:
    """

    ['Field name', 'Field description', 'Controlled value choices']

    :return:
    """
    df = pd.read_csv(DATA_PATH_INTENSITY_DATAFIELD_DESCRIPTIONS)
    return df


def iter_data() -> iter:
    yield 'japan', get_data_japan()
    yield 'kyoto', get_data_kyoto()
    yield 'liestal', get_data_liestal()
    yield 'meteoswiss', get_data_meteoswiss()
    yield 'south_korea', get_data_south_korea()
    yield 'washingtondc', get_data_washingtondc()


"""

    Our dataset

"""


DATA_PATH_AUGMENTED = f'{DATA_PATH}/augmented'
DATA_PATH_FULL = f'{DATA_PATH_AUGMENTED}/dataset.csv'


def get_dataset() -> pd.DataFrame:
    """

    :return:
    """

    df = pd.read_csv(DATA_PATH_FULL)
    return df


class TemperatureDataset(torch.utils.data.Dataset):

    def __init__(self):

        dataset = get_dataset()  # TODO -- normalize the values!

        dataset_original = pd.concat([df for _, df in list(iter_data())])
        dataset_original.set_index(['year', 'location'], inplace=True)

        self._bloom_data = dataset_original

        # self._bloom_data.apply(lambda x: (x - self._bloom_data['lat'].mean()) / self._bloom_data['lat'].std(), columns=['lat'])
        # self._bloom_data.apply(lambda x: (x - self._bloom_data['long'].mean()) / self._bloom_data['long'].std(), columns=['long'])
        # self._bloom_data.apply(lambda x: (x - self._bloom_data['alt'].mean()) / self._bloom_data['alt'].std(), columns=['alt'])

        data = []
        for i, df in dataset.groupby(['year', 'location']):
            df = df.loc[:, ['date', 'temp_max', 'temp_min', 'temp_mean']]
            df.set_index('date', inplace=True)
            df = df[~df.index.duplicated(keep='first')]  # TODO -- why are there duplicates??? remove in dataset!

            if len(df) == 366:
                df.drop(df.index[200], inplace=True)  # TODO -- remove which date?
            if len(df) != 365:
                print(len(df), i)
                continue

            data.append((i, df))

        self._temperature_data = data

    def __len__(self):
        return len(self._temperature_data)

    def __getitem__(self, index) -> dict:
        assert 0 <= index < len(self)

        i, df = self._temperature_data[index]

        year, location = i

        temp_max = torch.Tensor(df['temp_max'].values)
        temp_min = torch.Tensor(df['temp_min'].values)
        temp_mean = torch.Tensor(df['temp_mean'].values)

        entry = self._bloom_data.loc[year, location] # TODO -- stop  doing things in batches!

        bloom_doy = torch.Tensor([entry['bloom_doy'].values.mean()])  # TODO -- why are some bloom days stored twice??? -- remove .mean!!!
        lat = torch.Tensor([entry['lat'].values.mean()])
        lon = torch.Tensor([entry['long'].values.mean()])
        alt = torch.Tensor([entry['alt'].values.mean()])

        return {
            'year': year,
            'location': location,
            'dates': df.index.values,
            'temp_max': temp_max,
            'temp_min': temp_min,
            'temp_mean': temp_mean,
            'bloom_doy': bloom_doy,
            'lat': lat,
            'lon': lon,
            'alt': alt,
        }

    # def __getitem__(self, index) -> dict:
    #     if torch.is_tensor(index):
    #         index = index.tolist()
    #
    #     ixs = self._data.index.values[index]
    #
    #     items = self._data.iloc[index]
    #
    #     if isinstance(index, int):
    #         year, location = ixs
    #         date = items['date']
    #         temp_max = torch.Tensor([items['temp_max']])
    #         temp_min = torch.Tensor([items['temp_min']])
    #         temp_mean = torch.Tensor([items['temp_mean']])
    #     else:
    #         year = [year for year, location in ixs]
    #         location = [location for year, location in ixs]
    #         date = items['date'].values
    #         temp_max = torch.Tensor(items['temp_max'].values)
    #         temp_min = torch.Tensor(items['temp_min'].values)
    #         temp_mean = torch.Tensor(items['temp_mean'].values)
    #
    #     return {
    #         'year': year,
    #         'location': location,
    #         'date': date,
    #         'temp_max': temp_max,
    #         'temp_min': temp_min,
    #         'temp_mean': temp_mean,
    #     }


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random

    # _df = get_data_meteoswiss()
    # print(_df)
    # print(_df.columns)
    # print(_df.dtypes)
    # print(f'Number of locations: {_df["location"].nunique()}')
    #
    # # _df = _df.loc[_df['location'] == 'Japan/Wakkanai']
    # # _df = _df.loc[_df['location'] == _df.iloc[5000]['location']]
    #
    # fig, ax = plt.subplots()
    #
    # x = _df['year']
    # y = _df['bloom_doy']
    #
    # colors = {location: np.random.rand(3) for location in set(_df['location'].values)}
    #
    # ax.scatter(x, y, c=_df['location'].map(colors))
    #
    # ax.set_xlabel('year')
    # ax.set_ylabel('bloom day of year')

    # plt.show()

    # _df = get_dataset()
    #
    # print(_df)
    # print(_df.columns)

    _ds = TemperatureDataset()

    for _i in range(len(_ds)):
        print(_ds[_i])
    #     print(sorted(_ds[_i]['dates']))
    #     # print(_ds[_i]['temp_max'].shape)
        input()

    # _dl = torch.utils.data.DataLoader(_ds)
    #
    # for _i in _dl:
    #     print(_i)

    print(_ds)




