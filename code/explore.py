import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from mpl_toolkits.basemap import Basemap


import code.data as data


FIGURES_PATH = '../figures'


def plot_bloom_doy():

    for name, df in data.iter_data():

        path = f'{FIGURES_PATH}/bloom_doy/{name}'

        os.makedirs(path, exist_ok=True)

        for location, df_loc in df.groupby('location'):

            location = str(location).replace('/', '-')

            fig, ax = plt.subplots()

            x = df_loc['year']
            y = df_loc['bloom_doy']

            res = stats.linregress(x, y)

            pol = np.polyfit(x, y, 3)
            pol = np.poly1d(pol)

            ax.scatter(x, y,
                       s=1,
                       edgecolor='black',
                       )

            plt.plot(x, res.intercept + res.slope * x, label='linear fit')
            plt.plot(x, pol(x), label='poly fit')

            ax.set_xlabel('year')
            ax.set_ylabel('bloom day of year')
            plt.title(f'Bloom Day of Year over time for {location} ({name})')
            plt.legend()

            plt.savefig(f'{path}/{location}-bloom-doy.png')
            plt.cla()
            plt.close()


def show_map(lat, lon):

    max_lat, min_lat = max(lat), min(lat)
    max_lon, min_lon = max(lon), min(lon)

    map_height = 3000000
    map_width = 3000000

    map = Basemap(width=map_width,
                  height=map_height,
                  projection='lcc',
                  lat_0=lat.mean(),
                  lon_0=lon.mean(),
                  )

    map.scatter(lat.values, lon.values, latlon=True, marker='D')

    map.bluemarble()

    plt.show()


if __name__ == '__main__':


    # # # setup Lambert Conformal basemap.
    # # m = Basemap(width=12000000, height=9000000, projection='lcc',
    # #             resolution='c', lat_1=45., lat_2=55, lat_0=50, lon_0=-107.)
    # # # draw coastlines.
    # # m.drawcoastlines()
    # # # draw a boundary around the map, fill the background.
    # # # this background will end up being the ocean color, since
    # # # the continents will be drawn on top.
    # # m.drawmapboundary(fill_color='aqua')
    # # # fill continents, set lake color same as ocean color.
    # # m.fillcontinents(color='coral', lake_color='aqua')
    # # plt.show()
    #
    # df = data.get_data_japan()
    #
    # show_map(df['lat'], df['long'])
    #
    # pass

    # plot_bloom_doy()


    _df = pd.read_csv('data_augmented_sample.csv')

    print(_df)
    print(_df.columns)
    print(len(_df))
