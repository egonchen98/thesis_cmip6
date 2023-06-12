import datetime
import calendar
import math
import json
import pandas as pd
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings

import et0_calulator

warnings.simplefilter(action='ignore', category=FutureWarning)



def get_kc_ini_adjust(et0_mean, irri_interval, infiltration_depth=10):
    """2 dimensional interpolation with numpy
    et0_mean: mean et0 in the ini stage(mm)
    irri_interval: the day length after irrigation
    infiltration_depth: mm
    """
    x = [0, 2, 4, 6, 8, 10, 12]
    y = [1, 2, 4, 7, 10, 20]
    z_l = [
        [1.15, 1.15, 1.15, 1.15, 1.1, 1.02, 0.95],
        [1.15, 1.15, 1.15, 1.1, 1.05, 0.9, 0.85],
        [1.15, 1.15, 1.05, 0.9, 0.75, 0.65, 0.55],
        [1.15, 1.0, 0.75, 0.6, 0.5, 0.4, 0.35],
        [1.15, 0.8, 0.6, 0.45, 0.35, 0.3, 0.25],
        [1.15, 0.45, 0.3, 0.25, 0.15, 0.15, 0.1]
    ]

    z_s = [
        [1.15, 1.15, 1.1, 0.95, 0.9, 0.75, 0.7],
        [1.15, 1.15, 0.85, 0.7, 0.55, 0.5, 0.4],
        [1.15, 0.95, 0.55, 0.4, 0.3, 0.25, 0.2],
        [1.15, 0.7, 0.3, 0.25, 0.2, 0.15, 0.1],
        [1.15, 0.5, 0.25, 0.15, 0.15, 0.1, 0.1],
        [1.15, 0.25, 0.15, 0.1, 0.1, 0.05, 0.05]
    ]

    if infiltration_depth >= 40:
        z = z_l
        z_list= []
        for i in range(len(y)):
            zi = np.interp(et0_mean, x, z[i])
            z_list.append(zi)
        z_res = np.interp(irri_interval, y, z_list)
    elif infiltration_depth > 10:
        z1 = z_l
        z2 = z_s
        z1_list = []
        z2_list = []
        for i in range(len(y)):
            zi_1 = np.interp(et0_mean, x, z1[i])
            zi_2 = np.interp(et0_mean, x, z2[i])
            z1_list.append(zi_1)
            z2_list.append(zi_2)
        z_res1 = np.interp(irri_interval, y, z1_list)
        z_res2 = np.interp(irri_interval, y, z2_list)
        z_res = z_res2 + (infiltration_depth-10)/(40-10) * (z_res1 - z_res2)
    else:
        z = z_s
        z_list= []
        for i in range(len(y)):
            zi = np.interp(et0_mean, x, z[i])
            z_list.append(zi)
        z_res = np.interp(irri_interval, y, z_list)

    return z_res  # return Kc_ini

def get_kc_mid_end_adjusted(kc_tab, u2_mean, RHmin_mean, h_mean):
    """Get kc_mid or kc_end adjusted value during mid-season or late-season
    :u2: mean wind 2 meter in the mid stage
    :RHmin: mean RHmin in the mid stage
    :h: mean height of the crop in the mid stage
    :kc0: kc_mid_tab or kc_late_tab
    """
    if u2_mean<=6 and RHmin_mean<=80:
        return kc_tab + (0.04*u2_mean - 0.004*(RHmin_mean - 45)) * (h_mean/3)**0.3
    else:
        return kc_tab

def get_all_kc_by_curve(x: list, params: dict):
    """
    Get all kc values in all growth stage of a plant by linear interpolating.
    :x, list-like object that needed to be calculated (growth days)
    :kc_ini
    :kc_mid
    :kc_end
    :days_ini
    :days_dev
    :days_mid
    :days_late
    :return: a list of kc value matching x
    """
    stage_days = [0, params['days_ini'], params['days_dev'], params['days_mid'], params['days_late']]
    stage_days = np.cumsum(stage_days)
    kc_values = [params['kc_ini'], params['kc_ini'], params['kc_mid'], params['kc_mid'], params['kc_end']]
    points = {'X': stage_days, 'Y': kc_values}
    return np.interp(x, points['X'], points['Y'])

def get_kc_df(df_: pd.DataFrame, crop: dict):
    """Get kc value of a stage from a dataframe
    return kc series
    """
    crop = crop.copy()
    if len(df_) < sum([crop['days_ini'], crop['days_dev'], crop['days_mid'],
                       crop['days_late']]) - 1:  # not the whole growth stage
        return None
    # get kc_ini
    et0_mean_ini = df_.loc[df_.growth_stage == 'ini', 'et0'].mean()
    irri_interval = 20  # TODO: change the average interval in the future (now for arid area)
    kc_ini = get_kc_ini_adjust(et0_mean_ini, irri_interval)
    # get kc_mid
    u2_mean_mid, RHmin_mean_mid = df_.loc[df_.growth_stage == 'mid', ['u2', 'RHmin']].mean()
    h = crop['h']
    kc_mid = get_kc_mid_end_adjusted(crop['kc_mid'], u2_mean=u2_mean_mid, RHmin_mean=RHmin_mean_mid, h_mean=h)
    # get kc_end
    u2_mean_late, RHmin_mean_late = df_.loc[df_.growth_stage == 'late', ['u2', 'RHmin']].mean()
    kc_end = get_kc_mid_end_adjusted(crop['kc_end'], u2_mean_late, RHmin_mean_late, h)
    # get_all_kc_values matching date and add to the dataframe column
    crop['kc_ini'], crop['kc_mid'], crop['kc_end'] = kc_ini, kc_mid, kc_end
    kc_ser = pd.Series(get_all_kc_by_curve(df_['growth_day'], crop), df_.index)
    df_.loc[:, 'kc'] = kc_ser
    # return kc_ser
    return df_


class StationData:
    """Operations for Station data"""
    def __init__(self, origin_dir: Path, merged_dir: Path, preprocess_dir: Path, result_dir: Path):
        self.base_path = origin_dir
        self.merged_dir = merged_dir
        self.preprocess_dir = preprocess_dir
        self.result_dir = result_dir
        self.result_stations_dir = result_dir / 'stations'
        self.dataframe = None

    def merge_monthly_files(self):
        """Merge original monthly files of the same parameter
        write a merged pickle file ending with "_0.pkl" to datasets folder"""
        def add_col(var_name: str):
            """Generate names of columns"""
            common_col_names = ['station', 'latitude', 'longitude', 'altitude', 'year', 'month', 'day']
            prs_cols = ['prs_mean', 'prs_max', 'prs_min']
            prs_cols = prs_cols + [f'{i}_c' for i in prs_cols]
            tem_cols = ['tem_mean', 'tem_max', 'tem_min']
            tem_cols = tem_cols + [f'{i}_c' for i in tem_cols]
            rhu_cols = ['rhu_mean', 'rhu_min']
            rhu_cols += [f'{i}_c' for i in rhu_cols]
            pre_cols = ['pre_20_8', 'pre_8_20', 'pre_20_20']
            pre_cols += [f'{i}_c' for i in pre_cols]
            evp_cols = ['evp_small', 'evp_large']
            evp_cols += [f'{i}_c' for i in evp_cols]
            win_cols = ['win_mean', 'win_max', 'win_max_dir', 'win_big', 'win_big_dir']
            win_cols += [f'{i}_c' for i in win_cols]
            ssd_cols = ['ssd_hour']
            ssd_cols += [f'{i}_c' for i in ssd_cols]
            gst_cols = ['gst_mean', 'gst_max', 'gst_min']
            gst_cols = gst_cols + [f'{i}_c' for i in gst_cols]
            df_col_names = [common_col_names + i for i in [prs_cols, tem_cols, rhu_cols, pre_cols, evp_cols, win_cols, ssd_cols, gst_cols]]

            return [list_i for list_i in df_col_names if var_name.upper() in ''.join(list_i).upper()][0]

        def merge_files(folder):
            """Merge files with the same parameter"""
            suffix = 1
            if os.path.exists(f'{self.merged_dir}\\{folder}_{suffix}.pkl'):
                return None
            print(folder, end='/')
            files = [i for i in os.listdir(f'{self.base_path}\\{folder}') if i.endswith('.TXT')]
            col_names = add_col(f'{folder}')

            parameter_dfs = [pd.read_csv(f'{self.base_path}\\{folder}\\{file}', delim_whitespace=True, names=col_names) for file in files]
            print(f'{len(parameter_dfs)} files are read', end='/')
            merged_df = pd.concat(parameter_dfs).reset_index()
            print('concat', end='/')
            del parameter_dfs
            # merged_df.to_csv(f'{self.base_path}\\{folder}_{suffix}.csv')
            merged_df.to_pickle(f'{self.merged_dir}\\{folder}_{suffix}.pkl')
            del merged_df
            print(f'written.')

        pool = ThreadPoolExecutor(3)
        folders = [i for i in os.listdir(self.base_path) if os.path.isdir(f'{self.base_path}\\{i}')]
        for folder in folders:
            pool.submit(merge_files, folder)
        pool.shutdown(True)

    def preprocess_data(self, param):
        """数据预处理，针对过滤后的数据：
        0.5: 选取目标参数列
        1. 单位转换，转换为标准单位
        2. 合并年月日三列为一列
        3. 增加日序数列
        4. 增加弧度制纬度列
        :param: parameter name with 3 characters
        :return: dataframe
        """
        if Path(f'{self.preprocess_dir}/1{param}.pkl').exists():
            return None
        # df_ = pd.read_csv(f'{self.base_path}\\{param}_0.csv')
        df_ = pd.read_pickle(f'{self.merged_dir}\\{param}_1.pkl')
        # Select target columns
        common_cols = ['station', 'latitude', 'longitude', 'altitude', 'year', 'month', 'day']
        target_param = {'PRE': ['pre_20_20', 'pre_20_20_c'], 'RHU': ['rhu_mean', 'rhu_mean_c'],
                        'TEM': ['tem_max', 'tem_min', 'tem_mean', 'tem_max_c', 'tem_min_c', 'tem_mean_c'],
                        'WIN': ['win_mean', 'win_mean_c'], 'SSD': ['ssd_hour', 'ssd_hour_c']}

        df_ = df_[common_cols + target_param[param]]
        # Select rows with all quality code == 0
        code_cols = [i for i in target_param[param] if i.endswith('_c')]
        quality_condition = ' and '.join([f'{i}==0' for i in code_cols])
        df_ = df_.query(quality_condition)

        # df_.loc[df_.altitude > 1e5, 'altitude'] = df_.loc[df_.altitude > 1e5, 'altitude'] - 1e5  # altitude
        new_df_alt = df_.loc[df_.altitude > 1e5, 'altitude'] - 1e5  # altitude
        df_.update(new_df_alt)
        del new_df_alt

        if param == 'WIN':  # Process special value of wind
            # df_.loc[df_.win_mean > 300, 'win_mean'] = df_.loc[df_.win_mean > 300, 'win_mean'] - 300
            new_df_win = df_.loc[df_.win_mean > 300, 'win_mean'] - 300
            df_.update(new_df_win)
            del new_df_win

        if param == 'PRE':  # Process special value of precipitation
            new_df_pre = df_.loc[(3e4 < df_.pre_20_20) & (df_.pre_20_20 < 3.1e4), 'pre_20_20'] - 3e4  # 雪量特殊编码
            new_df_pre2 = df_.loc[(3.1e4 < df_.pre_20_20) & (df_.pre_20_20 < 3.2e4), 'pre_20_20'] - 3.1e4  # 雨雪总量特殊码
            df_.loc[df_.pre_20_20 > 3.2e4, 'pre_20_20'] = 0  # 微量或纯雾霜，视为0
            df_.update(new_df_pre)
            del new_df_pre
            df_.update(new_df_pre2)
            del new_df_pre2

        # 单位转换
        # df_['latitude'] = df_['latitude'] // 100 + df_['latitude'] % 100 / 60
        # df_['longitude'] = df_['longitude'] // 100 + df_['longitude'] % 100 / 60
        new_df_lat = df_['latitude'] // 100 + df_['latitude'] % 100 / 60
        new_df_lon = df_['longitude'] // 100 + df_['longitude'] % 100 / 60
        new_df_alt = df_['altitude'] / 10
        df_.update(new_df_lat)
        del new_df_lat
        df_.update(new_df_lon)
        del new_df_lon
        df_.update(new_df_alt)
        del new_df_alt

        if param != 'RHU':  # 除RHU外， 所有单位都需要除以0.1
            param_no_code = [i for i in target_param[param] if not i.endswith('_c')]
            # df_[param_no_code] = df_[param_no_code] / 10
            new_df_pc = df_[param_no_code] / 10
            df_.update(new_df_pc)
            del new_df_pc

        # 合粹年月日列并新增日序数列
        date_ser = df_['year'].astype('str') + '-' + df_['month'].astype('str') + '-' + df_['day'].astype('str')
        df_['date'] = pd.to_datetime(date_ser, infer_datetime_format=True, format='%Y-%m-%d')
        df_['cdays'] = df_['date'].dt.strftime('%j').astype('int32')
        # 增加弧度制纬度列
        df_['latitude_rad'] = df_['latitude'].apply(math.radians)
        df_ = df_[['station', 'latitude', 'longitude', 'altitude', 'latitude_rad', 'date', 'cdays'] +
                  [i for i in target_param[param] if not i.endswith('_c')]].round(2)

        # df_.to_csv(f'{self.preprocess_dir}\\1{param}.csv', index=False)
        df_.to_pickle(f'{self.preprocess_dir}\\1{param}.pkl')

    def merge_parameters(self):
        """Merge files of different parameters"""
        target_path = f'{self.preprocess_dir}\\merged_file.pkl'
        if os.path.exists(target_path):
            return None
        merge_on = ['station', 'latitude', 'longitude', 'altitude', 'latitude_rad', 'date', 'cdays']
        files = [i for i in os.listdir(self.preprocess_dir) if i.startswith('1')]

        df = pd.read_pickle(f'{self.preprocess_dir}\\{files[0]}')
        for index, file in tqdm(enumerate(files)):
            if index == 0:
                continue
            df_ = pd.read_pickle(f'{self.preprocess_dir}\\{file}')
            df = pd.merge(df, df_, on=merge_on, how='inner')
        df.rename(
            columns={'pre_20_20': 'tp', 'tem_mean': 'Tmean', 'tem_max': 'Tmax', 'tem_min': 'Tmin', 'rhu_mean': 'RH',
                     'ssd_hour': 'sunhour', 'win_mean': 'u2'}, inplace=True)
        df.drop(columns=['latitude_rad'], inplace=True)
        df.to_pickle(target_path)

    def write_origin_et0(self):
        """Calculate ET0 of the filtered file"""
        def write_et0_single_station(df_: pd.DataFrame, station: int):
            """Get et0 dataframe of a single station"""
            tar= self.result_stations_dir / f'{station}.pkl'
            if tar.exists():
                return None
            df_['Tmean_last'] = df_['Tmean'].shift(1).fillna(method='bfill')
            df_['delta_Tmean'] = df_['Tmean'] - df_['Tmean_last']
            df_[['RHmin', 'et0']] = df_.apply(lambda row: et0_calulator.get_et0_station(row), axis=1, result_type='expand')
            df_.to_pickle(tar)

        target_path = f'{self.result_dir}\\et0.pkl'
        if os.path.exists(target_path):
            return None

        df1 = pd.read_pickle(f'{self.preprocess_dir}\\merged_file.pkl')
        # df1 = df1.loc[(df1.date < '1962-01-01') & (df1.station < 51200)]

        stations = df1.station.unique()
        pool = ProcessPoolExecutor(max_workers=10)
        for station in stations:
            df_ = df1.loc[df1.station==station].copy()
            write_et0_single_station(df_, station)
            pool.submit(write_et0_single_station, df_, station)

        pool.shutdown()
        del df1

        dfs = [pd.read_pickle(self.result_stations_dir / f'{file}') for file in os.listdir(self.result_stations_dir)]
        df_et0_all = pd.concat(dfs).reset_index(drop=True)
        del dfs
        df_et0_all.to_pickle(self.result_dir / 'all_et0.pkl')

    def boundary_of_global_and_local_file______________________________________________functions(self):
        """----------------------------------------------------"""
        pass

    @staticmethod
    def get_filtered_area_df(df: pd.DataFrame, lat: list, lon: list):
        """Filter out target areas
        :lat: [st, end)
        :lon: [st, end]
        :return: dataframe
        """
        df = df.loc[(lat[0]<=df.latitude) & (df.latitude<=lat[1]) & (lon[0]<=df.longitude) & (df.longitude<=lon[1])]
        return df

    @staticmethod
    def get_filtered_growth_stage_df(df: pd.DataFrame, crop: dict):
        """filter out growth stage date"""
        df = df.copy()
        growth_stages = [0, crop['days_ini'], crop['days_dev'], crop['days_mid'], crop['days_late']]
        growth_days_cumsum = np.cumsum(growth_stages)
        st_date = datetime.date.fromisoformat(crop['start_date'])  # the first day
        end_date = st_date + datetime.timedelta(sum(growth_stages) - 1)  # the end day = st_date + delta_days - 1
        st_cdays = int(st_date.strftime('%j'))
        end_cdays = int(end_date.strftime('%j'))

        if end_date.year > st_date.year:
            df.loc[:, 'date1'] = df['date'] - datetime.timedelta(days=end_cdays)
            df.loc[:, 'crop_year'] = df['date1'].dt.year
            df.loc[:, 'is_leap'] = df['crop_year'].apply(calendar.isleap)
            df = df.loc[(df.cdays >= st_cdays) | (df.cdays <= end_cdays)]
            df.loc[:, 'growth_day'] = np.where(df['cdays'].values >= st_cdays, df['cdays'].values - st_cdays,
                                               df['cdays'].values + df['is_leap'].values + 365 - st_cdays)
            df.drop(columns=['date1', 'is_leap'], inplace=True)

        else:
            df.loc[:, 'crop_year'] = df['date'].dt.year
            df = df.loc[(df.cdays >= st_cdays) & (df.cdays <= end_cdays)]
            df.loc[:, 'growth_day'] = df['cdays'] - st_cdays

        map_keys = [df['growth_day'].isin(range(growth_days_cumsum[i], growth_days_cumsum[i + 1])) for i in
                    range(len(growth_days_cumsum) - 1)]
        map_values = ['ini', 'dev', 'mid', 'late']
        df.loc[:, 'growth_stage'] = np.select(map_keys, map_values)
        df.loc[:, 'growth_day'] = df['growth_day'] + 1
        return df

    @staticmethod
    def get_et0_pe_df(df: pd.DataFrame):
        """
        - effective rainfall
        - set values where et0<0 = 0
        return dataframe
        """

        df = df.copy()
        df.loc[:, 'et0'] = np.where(df['et0'].values<0, 0, df['et0'].values)
        tp_ser = df['tp'].values
        conditions = [tp_ser<=5, (5<tp_ser) & (tp_ser<=50), 50<tp_ser]
        operations = [0, 0.9, 0.75]
        df.loc[:, 'pe'] = np.select(conditions, operations)
        return df

    def write_target_et0(self, in_path: Path, out_path: Path, lat: list, lon: list, crop: dict, ):  # integrated function
        """Get target et0, pe from all eto file
        :in_path: all_et0
        :out_path: target_et0,pe path
        :lat: latitudes
        :lon: longitudes
        :crop: target_crop
        """
        if out_path.exists():
            return None
        (
            pd.read_pickle(in_path)
            .pipe(self.get_filtered_area_df, lat, lon)
            .pipe(self.get_filtered_growth_stage_df, crop)
            .pipe(self.get_et0_pe_df)
        ).reset_index(drop=True).to_pickle(out_path)

        return None

    @staticmethod
    def write_target_etc(in_path: Path, out_path: Path, crop: dict):
       """Get target etc/ir/idi dataframe
       :in_path: target_et0 path
       :crop: crop
       :return: dataframe
       """
       if out_path.exists():
           return None
       (
           pd.read_pickle(in_path)
           .groupby(by=['station', 'crop_year']).apply(get_kc_df, crop=crop)
           .reset_index(drop=True)
           .assign(
               etc=lambda x: x['et0'] * x['kc'],
               ir=lambda x: x['etc'] - x['pe'],
               idi=lambda x: x['ir'] / x['etc']
           )
           .to_pickle(out_path)
       )
       return None

    def get_cwdi_dekad(self, df: pd.DataFrame, filename: str):
        """
        Get CWDI for 10d data:
        - Get 10 days' accumulated etc and pe ( if no more than 10 days, use the mean value * 10)
        - Get cwdi0,  cwdi_1, cwdi_2, cwdi_3, cwdi_4 (set to zero if nan)
        - Get cwdi, coefficient = [0.3, 0.25, 0.2, 0.15, 0.1]
        - Get severity
        """
        file_path = self.result_dir / filename
        if file_path.exists():
            print(f'{filename} already exist.')
            return None

        df['year0'] = df['date'].dt.year
        df['month0'] = df['date'].dt.month
        # - Add Xu column to dataframe as a group label
        df.loc[df.date.dt.day <= 10, 'day0'] = 5
        df.loc[(10 < df.date.dt.day) & (df.date.dt.day <= 20), 'day0'] = 15
        df.loc[df.date.dt.day > 20, 'day0'] = 25
        df.drop(columns=['date', 'stage'], inplace=True)

        # - Get mean value of 10 days
        df = df.groupby(by=['station', 'year0', 'month0', 'day0']).mean()  # TODO: Change to sum etc/tp, but keep other parameters mean.
        df['cwdi_0'] = (1 - df['pe'] / df['etc']) * 100
        cwdi_columns = [f'cwdi_{i}' for i in [1, 2, 3, 4]]
        for i in [1, 2, 3, 4]:
            df[cwdi_columns[i - 1]] = df['cwdi_0'].shift(i).fillna(method='bfill')

        df['cwdi'] = df['cwdi_0'] * 1 + df['cwdi_1'] * 0.3 + df['cwdi_2'] * 0.25 + df['cwdi_3'] * 0.15 + df[
            'cwdi_4'] * 0.1

        df.to_pickle(self.result_dir / filename)
        return df

    def run(self):
        """Run main code"""
        # self.merge_monthly_files()  # Merge files
        # print('- All parameters are merged.')
        # # preprocess data
        # params = ['PRE', 'TEM', 'WIN', 'SSD', 'RHU']
        # pool2 = ProcessPoolExecutor(3)
        # for param in params:
        #     pool2.submit(self.preprocess_data, param)
        # pool2.shutdown(True)
        # print('- All parameters are preprocessed.')
        #
        # self.merge_parameters()  # Merge parameters
        # print(f'- All parameters are merged.')
        # self.write_et0()

        # ==== specially, for arid area ====

        crop_type = 'arid_spring_maize'  # get etc
        crop = json.loads(Path('../resources/crop_property.json').read_text())[crop_type]
        et0_path = self.result_dir / 'all_et0.pkl'
        tar_et0 = self.result_dir / f'{crop_type}_et0.pkl'
        tar_etc = self.result_dir / f'{crop_type}_etc.pkl'
        lat = [31.69, 49.24]
        lon = [75, 111.41]
        st_time = time.time()
        self.write_target_et0(in_path=et0_path, out_path=tar_et0, lat=lat, lon=lon, crop=crop)  # write the pickle file
        t1 = time.time() - st_time
        print(t1)
        self.write_target_etc(in_path=tar_et0, out_path=tar_etc, crop=crop)
        print(time.time()-st_time)






if __name__ == '__main__':
    config = json.loads(Path('../resources/config.json').read_text(encoding='utf8'))

    sd = StationData(origin_dir=Path(config['origin_data_dir']), merged_dir=Path(config['merged_data_dir']),
                     preprocess_dir=Path(config['preprocess_data_dir']), result_dir=Path(config['result_data_dir']))
    sd.run()