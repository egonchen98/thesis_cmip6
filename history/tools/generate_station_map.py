import folium
from folium.plugins import Geocoder
import pandas as pd


def draw_map(station_df):
    """Draw interactive geomap with meteorological station"""
    # station_df['latitude'] = station_df['latitude'] // 100 + station_df['latitude'] % 100 / 60
    # station_df['longitude'] = station_df['longitude'] // 100 + station_df['longitude'] % 100 / 60
    station_df['station'] = station_df['station'].astype('int32')
    # station_df = station_df.drop_duplicates(keep='last')
    station_df = station_df.round(2)
    TILES = ['cartodbpositron', 'http://map.geoq.cn/ArcGIS/rest/services/ChinaOnlineStreetGray/MapServer/tile/{z}/{y}/{x}']  # Map Theme
    m = folium.Map(location=[station_df['latitude'].mean(), station_df['longitude'].mean()], zoom_start=4, tiles=TILES[1], attr='灰色版')  # add a map
    # Add markers
    for index, row in station_df.iterrows():
        folium.Circle(
            location=(row['latitude'], row['longitude']),
            popup=f'{row["station"]}\n{row["latitude"]}, {row["longitude"]}',
            radius=6,
            color='black'
        ).add_to(m)
        if index == 3:
            break
    # Add a search bar
    Geocoder().add_to(m)
    # Save html
    m.save('../resources/station_geomap.html')
    return m


if __name__ == '__main__':
    df1 = pd.read_pickle(
        r'F:\cy\Database\ChinaStation\中国国家级地面气象站基本气象要素日值数据集（V3.0）SURF_CLI_CHN_MUL_DAY_V3.0\datasets\2_preprocess_data\merged_file3.pkl')
    df2 = df1.loc[df1.cdays == 1][['station', 'latitude', 'longitude', 'date']]
    df2 = df2.loc[df2.date.dt.year == 1960]
    draw_map(df2)
