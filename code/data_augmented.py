import datetime
import time

from tqdm import tqdm

import numpy as np
import pandas as pd

from pydap.client import open_url
from pydap.cas.urs import setup_session

import code.data as data


def iter_dates():
    for year in range(1980, 2022):
        for month in range(1, 12 + 1):
            num_days = (datetime.date(year + (month // 12), (month % 12) + 1, 1) - datetime.date(year, month, 1)).days
            for day in range(1, num_days + 1):
                date = datetime.date(year, month, day)
                yield date


def iter_dates_urls():
    for date in iter_dates():
        year = date.year
        month = date.month

        # Substitute the date in the url to know where we can obtain the data
        # Each date has its own source
        # Also some number changes after 1991, 2000 and 2010 and I could not be bothered to figure out why
        weird_number = 100
        if year >= 1992:
            weird_number += 100
        if year >= 2001:
            weird_number += 100
        if year >= 2011:
            weird_number += 100
        url = f'https://goldsmr5.gesdisc.eosdis.nasa.gov/' \
              f'opendap/hyrax/MERRA2/M2I3NVASM.5.12.4/' \
              f'{year}/{month:02}/MERRA2_{weird_number}.inst3_3d_asm_Nv.{date.strftime("%Y%m%d")}.nc4'

        yield date, url


def get_credentials() -> tuple:
    # Obtain credentials to access the data
    # A credentials file is simply a username and password separated by a single space
    with open('credentials.txt', 'r') as f:
        tokens = f.read().split(' ')
        username = tokens[0]
        password = ' '.join(tokens[1:])
    return username, password


def augment_data():
    """
    Take the original datasets provided in the Cherry Blossom Prediction Competition and augment it with temperature
    data from the MERRA-2 dataset: https://disc.gsfc.nasa.gov/datasets/M2T3NPTDT_5.12.4/summary

    The MERRA-2 dataset contains daily 3-hour-interval measurements of temperature around the globe. The dataset
    is queried for the locations that are relevant to the competition.

    MERRA-2 data specification:
    https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf

    """

    # Create a dataset object that will contain the augmented data
    dataset = pd.DataFrame(
        columns=['year',
                 'date',
                 'original_dataset',
                 'location',
                 'lat',
                 'lon',
                 'alt',
                 'temps',
                 'temp_max',
                 'temp_min',
                 'temp_mean',
                 ]
    )

    """
        Obtain temperature data
    """

    # Obtain credentials to access the data
    username, password = get_credentials()

    dates_urls = list(iter_dates_urls())

    url_iter = tqdm(dates_urls,
                    total=len(dates_urls),
                    desc='Obtaining MERRA-2 data: ',
                    )

    for i, (date, url) in enumerate(url_iter):

        # Connect to the data source
        session = setup_session(username=username, password=password, check_url=url)
        data_connection = open_url(url, session=session)

        # The temperature data is stored in a grid corresponding to the latitude and longitudes of the
        # measurement points.
        # Obtain a 'grid' of how the latitude and longitude are mapped to positions in the grid
        lat_vals = np.array(data_connection['lat'][:].data)
        lon_vals = np.array(data_connection['lon'][:].data)

        # Obtain the entire grid of temperature data for all latitudes/longitudes
        # Note: this is much faster than making separate requests for only relevant latitudes/longitudes
        # The dataset is stored in a 4-dimensional tensor where the axes correspond to:
        #   - the (8) 3-hourly temperature measurements in the day
        #   - the pressure level at which this measurement was made
        #     this corresponds to the altitude of the measurement and is divided in 72 'levels'
        #     level 72 corresponds to 'surface temperature'
        #     source: http://wiki.seas.harvard.edu/geos-chem/index.php/MERRA-2
        #   - the latitude
        #   - the longitude
        temps_all = data_connection['T'][:, 71, :, :]
        # Cast the data to a numpy array and convert from Kelvin to Celsius
        temps_all = np.array(temps_all) - 272.15
        # Remove the altitude dimension
        temps_all = temps_all.squeeze()
        # Move the number-of-temperature-measurements-axis to the end
        temps_all = np.moveaxis(temps_all, 0, -1)

        # Process all datasets provided by the competition
        for name, df in data.iter_data():

            # Filter the dataset for entries that correspond to the year that we currently look at
            df_year = df.loc[df['year'] == date.year]

            # For all latitudes in the dataset, get the closest measurement point index in MERRA-2
            lat_ixs = [int((np.abs(lat_vals - lat)).argmin()) for lat in df_year['lat']]
            # For all longitudes in the dataset, get the closest measurement point index in MERRA-2
            lon_ixs = [int((np.abs(lon_vals - lon)).argmin()) for lon in df_year['long']]

            # Get the temperatures that are relevant to the competition dataset
            temps = temps_all[lat_ixs, lon_ixs, :]

            # Create new data entries
            for (_, row), temp in zip(df_year.iterrows(), temps):
                entry = {
                    'year': date.year,
                    # 'date': date.strftime('%Y-%m-%d'),
                    'date': date,
                    'original_dataset': name,
                    'location': row['location'],
                    'lat': row['lat'],
                    'lon': row['long'],
                    'alt': row['alt'],
                    'temps': temp,
                    'temp_max': max(temp),
                    'temp_min': min(temp),
                    'temp_mean': temp.mean(),
                }

                dataset.loc[len(dataset)] = entry

        if not i % 10000:
            dataset.to_csv('data_augmented_checkpoint.csv', index=False)

    dataset.to_csv(f'data_augmented.csv', index=False)


if __name__ == '__main__':
    # with open('credentials.txt', 'r') as f:
    #     tokens = f.read().split(' ')
    #     username = tokens[0]
    #     password = ' '.join(tokens[1:])
    #
    #
    # from pydap.client import open_url
    # from pydap.cas.urs import setup_session
    #
    # # url = 'https://gpm1.gesdisc.eosdis.nasa.gov/dods/GPM_3IMERGHHL_06'
    # # url = 'https://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/MERRA2_DIURNAL/'
    # url = 'https://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/hyrax/MERRA2/M2I3NVASM.5.12.4/1980/01/MERRA2_100.inst3_3d_asm_Nv.19800101.nc4'
    # session = setup_session(username=username, password=password, check_url=url)
    # dataset = open_url(url, session=session)
    #
    # # print(dataset['T'][:, 0, 0])
    #
    # lat, lon = 0.00, 0.00
    #
    # lat_vals = dataset['lat'][:].data
    # lon_vals = dataset['lon'][:].data
    #
    # print(len(lat_vals))
    # print(len(lon_vals))
    #
    # # (np.abs(array - value)).argmin()
    #
    # lat_indices = np.where(
    #     (lat_vals >= lat) &
    #     (lat_vals <= lat + 2)
    # )[0]
    # lon_indices = np.where(
    #     (lon_vals >= lon) &
    #     (lon_vals <= lon + 2)
    # )[0]
    #
    # lat_idx = int(lat_indices[0])
    # lon_idx = int(lon_indices[0])
    #
    # print(lat_idx)
    # print(lon_idx)
    #
    # data = dataset['T'][:, 72 - 1, lat_idx, lon_idx]
    #
    # import numpy as np
    #
    # data = np.array(data)
    #
    # print(data.shape)
    # print(data[0])

    augment_data()
