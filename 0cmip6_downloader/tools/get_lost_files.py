import os
import shutil
from pathlib import Path
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


def get_downloaded_files(root_paths: list) -> pd.DataFrame:
    """Get a dataframe logging information of downloaded files"""
    file_list = []
    for dir_path in root_paths:
        for (root, dirs, files) in os.walk(dir_path):
            for file in files:
                file_path = f'{root}\\{file}'
                filename = file
                file_list.append([file_path, filename])

    file_df = pd.DataFrame(file_list, columns=['file_path', 'filename'])
    return file_df


def filter_files(df_all: pd.DataFrame, df_exist: pd.DataFrame) -> pd.DataFrame:
    """Filter files that are not downloaded"""
    df_merged = pd.merge(df_all, df_exist, on='filename', how='left')
    df_merged.drop_duplicates(subset='filename', inplace=True)
    df_lost = df_merged.loc[df_merged.file_path.isnull()]
    return df_lost


def move_files_to_database(source_dir: Path, target_dir: Path):
    """Remove files in the source directory to the target dir"""
    file_df = get_downloaded_files(source_dir)
    pool = ThreadPoolExecutor(max_workers=10)
    for index, row in file_df.iterrows():
        source_path = row['file_path']
        filename = row['filename']
        if not filename.endswith('.nc'):
            continue
        if (Path(target_dir) / Path(filename)).exists():
            continue

        print(f'Moving {row["filename"]}')
        shutil.move(source_path, target_dir/Path(row['filename']))
        if index % 1000 == 0:
            print(index)


def remove_exist_files(source_dir: Path, target_dir: Path):
    """Remove duplicated files in source dir"""
    files_df = get_downloaded_files(source_dir)
    for index, row in files_df.iterrows():
        filename = row['filename']
        source_path = row['file_path']
        target_path = target_dir / Path(filename)
        if target_path.exists():
            os.remove(source_path)

            print(filename)


def run():
    """Run"""
    cur_dir = os.path.dirname(__file__)
    root_dir = os.path.dirname(cur_dir)
    config_path = Path(root_dir) / 'resources/config.json'
    config = json.loads(config_path.read_text())
    all_file_csv_path = Path(f'../resources/all_files.csv')

    dir_paths = [config['all_database'], Path(r'I:\cmip'), Path(r'F:\cy\Database\CMIP6\0_origin')]
    df_exist = get_downloaded_files(dir_paths)
    df_all = pd.read_csv(all_file_csv_path)

    lost_df = filter_files(df_all, df_exist)
    lost_df.to_csv('../resources/lost.csv', index=False)
    return lost_df


if __name__ == '__main__':
    run()
    exit()
    config_path =Path('../resources/config.json')
    config = json.loads(config_path.read_text())
    all_file_csv_path = Path(f'../resources/all_files.csv')
    download_dir1 = config['download_database']
    target_dir = config['all_database']

    source_dir = Path(download_dir1)
    target_dir = Path(target_dir)
    # remove_exist_files(source_dir, target_dir)
    move_files_to_database(source_dir, target_dir)