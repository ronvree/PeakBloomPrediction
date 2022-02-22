import numpy as np
import pandas as pd

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random

    _df = get_data_meteoswiss()
    print(_df)
    print(_df.columns)
    print(_df.dtypes)
    print(f'Number of locations: {_df["location"].nunique()}')

    # _df = _df.loc[_df['location'] == 'Japan/Wakkanai']
    # _df = _df.loc[_df['location'] == _df.iloc[5000]['location']]

    fig, ax = plt.subplots()

    x = _df['year']
    y = _df['bloom_doy']

    colors = {location: np.random.rand(3) for location in set(_df['location'].values)}

    ax.scatter(x, y, c=_df['location'].map(colors))

    ax.set_xlabel('year')
    ax.set_ylabel('bloom day of year')

    # plt.show()

    _df = get_status_intensity_datafield_descriptions()

    print(_df)
    print(_df.columns)




