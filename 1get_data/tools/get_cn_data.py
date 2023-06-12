import os
import json
from joblib import Parallel, delayed

import xarray as xr
import pandas as pd
import time
import datetime
from pathlib import Path


def write_area_nc(in_path: Path, out_path: Path, lat: list, lon: list):
    """
    Filter nc data and keep only China data.
    :in_path: input nc file path
    :out path: output nc file path
    :lat: latitudes
    :lon: longitudes
    """
    if out_path.exists():
        return None
    try:
        (
            xr.open_dataset(in_path)
            .sel(lat=slice(lat[0], lat[1]), lon=slice(lon[0], lon[1]))
            .to_netcdf(out_path)
        )
    except Exception as e:
        print('eorror: ', e)
    finally:
        return None


def run():
    st_time = time.time()
    config = json.loads(Path('../resources/config.json').read_text())
    all_origin_files = [file for file in os.listdir(config['origin_database']) if file.endswith('.nc')]
    origin_database = Path(config['origin_database'])
    cn_nc_database = Path(config['cn_nc_database'])
    lat = [18, 54]
    lon = [73, 135]

    # Get cn nc files
    Parallel(n_jobs=-1)(delayed(write_area_nc)(origin_database/file, cn_nc_database/file, lat, lon) for file in all_origin_files)

    print(time.time() - st_time)


if __name__ == '__main__':
    run()