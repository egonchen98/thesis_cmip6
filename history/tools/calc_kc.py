import math
import datetime
from itertools import accumulate
import numpy as np




def get_CWDI_i(params: dict):
    """get 10-day-CWDI
    :tp: cumulative precipitation
    :etc: crop evapotranspiration
    :return: CWDI_i
    """
    CWDI_i = (1- params['etc'] / params['tp']) * 100 if params['etc'] > params['tp'] else 0
    return CWDI_i

def get_CWDI(params: dict):
    """Get CWDI
    :CWDI, CWDI_i1, CWDI_i2, CWDI_i3, CWDI_i4
    :return: CWDI"""
    CWDI = params['CWDI_i'] * 0.3 + params['CWDI_i1'] * 0.25 + params['CWDI_i2'] * 0.2 + params['CWDI_i3'] * 0.15 + params['CWDI_i4'] * 0.1
    return CWDI



def get_crop_property():
    """Return crop properties"""
    base_crops = {
        'rice': {
            'start_date': '2024-06-15',
            'growth_stage_length': [30, 40, 55, 25],  # L_ini, L_dev, L_mid, L_late
            'kc': [1.05, 1.56, 0.75]  # 海宁为例，数据库中存的数据
        },
        'south_wheat': {
            'start_date': '2024-11-13',
            'growth_stage_length': [103, 17, 46, 25],  # TODO 核实南方小麦生长期，Init有103天太多了?
            'kc': [0.8, 1.6, 0.95]
        },
        'maize': {
            'start_date': '2024-06-11',
            'growth_stage_length': [20, 40, 30, 13],  # ref 杨凌地区， 陈， 2006
            'kc': [0.15, 1.15, 0.5]
        },
        'north_wheat': {
            'start_date': '2024-10-11',
            'growth_stage_length': [130, 50, 45, 16],  # ref 杨凌地区， 陈， 2006
            'kc': [0.15, 1.10, 0.15]
        }
    }

    return base_crops


if __name__ == '__main__':
    crop = 'south_wheat'
    date = '2024-03-10'
    kc = get_kc(crop, date)
    print(kc)