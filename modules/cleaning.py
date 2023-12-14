import pandas as pd
import math
import numpy as np
import sensormotion as sm
import re

columns_whitelist = ['timestamp', 'label', 'subject', 'session']


def remove_data_errors(
        df: [pd.DataFrame],
        error_sub_method: str = 'mean'
) -> ([pd.DataFrame], int):
    """
    Remove NaNs or infinite values on the dataset
    :param df: input dataset
    :param error_sub_method: the technique to apply in order to clean the dataset
    :return: cleaned dataset
    """
    total_errors_found = 0
    
    for df_i in df:
        for column in df_i:
            if column.split("_")[0] == "data":
                # regex for numbers and dots -> "2.5" (positive and negative)
                nums = re.compile(r"-?\d+(?:\.\d+)?")
                df_i[column] = [round(float(nums.findall(x)[0]), 2) if type(x)==str else x for x in df_i[column]]

        # Handle Data Errors: clean from NaN and infinite values
        df_i.replace([math.inf, -math.inf], np.nan, inplace=True)
        if df_i.isna().sum().sum() > 0:
            total_errors_found += df_i.isna().sum().sum()
            # Apply correction
            if error_sub_method == 'mean':
                df_i.fillna(df_i.mean(), inplace=True)

            elif error_sub_method == 'forward':
                df_i.fillna(method='ffill', inplace=True)

            elif error_sub_method == 'backward':
                df_i.fillna(method='bfill', inplace=True)

            else:
                pass
            
    return df, total_errors_found


def remove_noise(
        df: [pd.DataFrame],
        sample_rate: int,
        filter_name: str,
        cutoff: int or tuple,
        order: int = 4
) -> [pd.DataFrame]:
    """
    Apply filtering to remove noisy data
    :param df: the dataset
    :param sample_rate: sample frequency of the recorded dataset in Hz
    :param filter_name: name of the filter to be applied, it can be {'lowpass', 'highpass', 'bandpass'}
    :param cutoff: the cutoff frequency for the filter.
    :param order: order of the filter.
    :return: filtered array of datasets
    """
    # Instantiate the filter
    b, a = sm.signal.build_filter(frequency=cutoff,
                                  sample_rate=sample_rate,
                                  filter_type=filter_name,
                                  filter_order=order)

    for df_i in df:
        for column in df_i.columns:
            if column not in columns_whitelist:
                df_i[column] = sm.signal.filter_signal(b, a, signal=df_i[column].values)

    return df
