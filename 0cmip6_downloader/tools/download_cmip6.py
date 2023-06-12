import datetime
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
import json
from joblib import Parallel, delayed

import requests
import pandas as pd
from pathlib import Path

import wget
from selenium.webdriver.firefox.service import Service
from selenium import webdriver


def get_wget_files() -> None:
    """Get(download) wget.sh from a html file using selenium
    1. get every item
    2. click on 'get wget'
    3. wait a second to click another one and loop."""
    driver = webdriver.Firefox(service=Service(executable_path=r'D:\OneDrive - HHU\MainSync\03_Life\02_Self\scripts\geckodriver.exe'))
    driver.get('file:///E:/CMIP6 Training/code/resources/download_links/cart.html')
    try:
        driver.implicitly_wait(1)
        elements = driver.find_elements(by='partial link text', value='WGET Script')
        index = 0
        for elem in elements[1:]:
            index = index + 1
            elem.click()
            driver.implicitly_wait(1)
            if index % 10 == 0:
                print(index)
    except Exception as e:
        print(e)
    finally:
        driver.close()

def get_all_url(wget_file_folder: Path) -> pd.DataFrame:
    """Get download url from wget.sh files"""
    dfs = []
    for file in os.listdir(wget_file_folder):
        if not file.endswith('.sh'):
            continue
        path = Path(f'{wget_file_folder}/{file}')
        text = path.read_text()
        target_text = re.findall("'http://.*nc'", text, flags=re.MULTILINE)
        df = pd.DataFrame(target_text, columns=['url'])
        df['place'] = file
        df['filename'] = df['url'].str.split('/').apply(lambda row: row[-1])
        dfs.append(df)

    df = pd.concat(dfs)
    df['url'] = df['url'].str.strip("'")
    df['filename'] = df['filename'].str.strip("'")
    df[['param', 'frequency', 'model', 'scenario', 'r1i1p1f1', 'gn', 'date_range']] = \
        df['filename'].str.split('_', expand=True)
    df[['st_date', 'end_date']] = df['date_range'].str.split('-', expand=True)
    df[['end_date', 'file_type']] = df['end_date'].str.split('.', expand=True)
    df.drop(columns=['frequency', 'r1i1p1f1', 'date_range', 'file_type'], inplace=True)
    df.dropna(how='any', axis=0, inplace=True)
    df[['st_date', 'end_date']] = df[['st_date', 'end_date']].astype('int32')
    df = df.loc[~((df.end_date < 19510101) | (df.st_date > 21010101))]

    return df

def get_exist_files(folder: Path) -> pd.DataFrame:
    """get downloaded files
    return dataframe
    """
    files = [file for file in os.listdir(folder) if file.endswith('.nc')]
    df_files = pd.DataFrame(files, columns=['filename'])
    df_files['exist'] = 1
    return df_files

def download_with_wget(url: str, tar_path: Path) -> None:
    """Download file with wget library"""
    if os.path.exists(tar_path):
        return None
    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}--- downloading {tar_path}...')
    wget.download(url, tar_path)


def run():
    """run"""
    config = json.loads(Path('../resources/config.json').read_text())
    wget_file_folder = '../resources/wget_files'
    # origin_folder = Path(config['all_database'])
    origin_folder = Path('../resources/wget_files')
    cn_folder = config['cn_nc_database']
    df_all = get_all_url(wget_file_folder)
    df_exist = get_exist_files(cn_folder)
    df_lost_files = pd.merge(df_all, df_exist, on='filename', how='left')
    df_lost_files = df_lost_files.loc[df_lost_files.exist!=1, :]
    df_lost_files.to_csv('../resources/lost_files.csv')

    # for index, row in df_lost_files.iterrows():
    #     print(row['url'])
    #     download_with_wget(row['url'], '../resources/wget_files/' + row['filename'])


if __name__ == '__main__':
    run()
