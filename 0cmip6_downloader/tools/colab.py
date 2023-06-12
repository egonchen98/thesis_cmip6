import pathlib
import random
import wget
import datetime
import requests

import sys
import pandas as pd
import pymysql
from sqlalchemy import create_engine
import cProfile
import os
import json
from pathlib import Path


def write_df(df: pd.DataFrame) -> None:
    """Write file table to mysql"""
    conn = create_engine('mysql+pymysql://colab:colab123456@124.220.27.50/colab')

    df.to_sql('cmip6_files', conn, if_exists='replace', index=False)

    return None


def get_1_record():
    """Get one record from mysql database"""
    conn = pymysql.connect(host='124.220.27.50', port=3306, user='colab', password='colab123456', db='colab')
    read_sql = "SELECT * FROM `colab`.`cmip6_files` WHERE `status` = 'not requested' LIMIT 0,100"
    get_col_name_sql = "select COLUMN_NAME from INFORMATION_SCHEMA.COLUMNS where TABLE_NAME='cmip6_files'"
    with conn:
        with conn.cursor() as cur:
            # Get column names
            cur.execute(get_col_name_sql)
            columns = cur.fetchall()
            columns = [i[0] for i in columns]
            # Get a record
            cur.execute(read_sql)
            res = cur.fetchall()
            if len(res) == 0:
                return 'Finished'
            res = random.choice(res)
            res = dict(zip(columns, res))
            # Update value of the record
            log_downloading_sql = f"UPDATE `colab`.`cmip6_files` SET `status` = 'downloading' WHERE `url` = '{res['url']}'"
            cur.execute(log_downloading_sql)
        conn.commit()

    return res


def download_1_record(res: dict, data_folder: str) -> None:
    """Download a record with wget library"""

    file_folder = f'{data_folder}'
    Path(file_folder).mkdir(parents=True, exist_ok=True)
    file_path = f'{file_folder}\\{res["filename"]}'

    # Check file existence
    if not os.path.exists(file_path):
        # Check url connection with requests library
        status_code = requests.head(res['url']).status_code
        if status_code != 200:
            return None
        # Download file with wget library
        print(f'{datetime.datetime.now().strftime("%m-%d %H:%M:%S")}--- downloading {res["filename"]}')
        wget.download(res['url'], file_path)

    return None


def update_downloaded_record(res: dict) -> None:
    """Update status of a record in mysql database"""
    conn = pymysql.connect(host='124.220.27.50', port=3306, user='colab', password='colab123456', db='colab')
    update_sql = f"UPDATE `colab`.`cmip6_files` SET `status` = 'downloaded' WHERE `url` = '{res['url']}'"
    with conn:
        with conn.cursor() as cur:
            cur.execute(update_sql)
        conn.commit()
    return None


def get_existing_logs(data_folder: str) -> pd.DataFrame:
    """Get existing files ending with .nc under the data_folder directory containing childer folder"""
    filelist = []
    # Get all files under the database folder
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.nc'):
                filelist.append(file)

    df1 = pd.DataFrame(filelist, columns=['filename'])
    df0 = pd.read_csv('../resources/all_files.csv')
    df_merge = pd.merge(df0, df1, on='filename', how='right')
    df_merge['status'] = 'downloaded'
    return df_merge


def run(res: dict, database: 'str') -> str:
    """Run"""
    try:
        download_1_record(res=res, data_folder=database)
        update_downloaded_record(res=res)
    except Exception as e:
        print('Some Error Occurred: ', e)
    return 'continue'


if __name__ == '__main__':
    # reset database in mysql
    df = pd.read_csv('../resources/lost_files.csv')
    df['status'] = 'not requested'
    write_df(df=df)

