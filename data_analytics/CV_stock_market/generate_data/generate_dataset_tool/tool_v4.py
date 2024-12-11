import sys
sys.path.append("../../")

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pyspark import  SparkContext
from pyspark.sql import SparkSession
from ta import add_all_ta_features
from ta.utils import dropna
from utils.draw_candle_image import *
from generate_data.query_dataset import *
import json
import keras
import numpy as np
import numpy as np
import os
import pandas as pd
import random
import ta
import ta.momentum
import ta.trend
import ta.trend
import ta.volatility
import tensorflow as tf
from tabulate import tabulate


def generate_three_image_of_input_with_ema_9_macd_histogram(
    df_date: pd.DataFrame, 
    df_original: pd.DataFrame, 
    list_previous_days: int, 
    list_next_days: int,
    days_result: int,
    figscale: int,
    indicator = {}
    ):
    
    list_images_3_days = []
    list_images_7_days = []
    list_images_30_days = []
    list_prices = []
    list_dates = []
    list_ema_9 = []
    list_macd_histogram = []
    
    if len(df_date) == 0:
        print(f"Total: 0 images")
        return (
            list_images_3_days,
            list_images_7_days,
            list_images_30_days, 
            list_prices, 
            list_dates, 
            list_ema_9, 
            list_macd_histogram)
    
    count = 0
    total_images = df_date[Total_records].values[0]
    print(f"Total: {total_images} images")
    
    df_original = convert_number_index(df_original)
    
    for date in df_date[Date_normalized]:
        
        count += 1
        if count % 100 == 0:
            print(f"Load {count}/{total_images} images")
        
        is_save_this_day = True
        for previous_days, next_days in zip(list_previous_days, list_next_days):
            if not is_save_this_day:
                continue
            
            total_days = previous_days + 1 + next_days + days_result
            
            draw_df_pandas = add_days_around_date(date, df_original, previous_days, next_days)
            if len(draw_df_pandas) < (previous_days + next_days):
                is_save_this_day = False
                continue
            
            image_tensor = draw_candle_image(
                draw_df_pandas, 
                show_x_y=False, 
                show_volume=False, 
                
                **indicator,
                
                figscale=figscale, 
                figcolor="black",
                return_image_tensor=True
            )
            
            image_tensor = normalize_candle_image(
                image_tensor, 
                combine_into_one_image=False,
                convert_red=False,
                convert_green=False,
            )
            
            date = normalize_date(date)
            index_of_date = get_date_index(df_original, date)
            start_index = index_of_date - previous_days
            end_index = index_of_date + next_days + days_result
            data_result_df = df_original[start_index:end_index + 1]
            
            if 0 < len(data_result_df) < total_days:
                data_result_df = duplicate_last_row(data_result_df, total_days - len(data_result_df))
            elif len(data_result_df) == 0:
                is_save_this_day = False
                continue
            
            targets = data_result_df[[High, Open, Close, Low]].to_numpy()
            ema_9 = data_result_df[[EMA_9]].to_numpy()
            macd_history = data_result_df[[Macd_histogram]].to_numpy()
            
            # trend_type_df = df_date[df_date[Date_normalized] == date_str][[Group_trend_3_day_type]]
            # trend_type = trend_type_df.values[0][0]
            
            if previous_days == 30:
                list_images_30_days.append(image_tensor)
                list_prices.append(targets)
                list_dates.append(str(date))
                list_ema_9.append(ema_9)
                list_macd_histogram.append(macd_history)
            elif previous_days == 7:
                list_images_7_days.append(image_tensor)
            elif previous_days == 0:
                list_images_3_days.append(image_tensor)
    
    return (
        list_images_3_days,
        list_images_7_days,
        list_images_30_days, 
        list_prices, 
        list_dates, 
        list_ema_9, 
        list_macd_histogram)


def save_to_tensorflow_dataset_with_ema_9_macd_histogram(
    df_date: pd.DataFrame, 
    df_original: pd.DataFrame, 
    previous_days: int, 
    next_days: int,
    days_result: int,
    figscale: int,
    number_images_preview = 0,
    save_dataset_to_folder = None,
    number_images_to_save = None,
    indicator = {},
    function_generate = None,
    list_previous_days = [30, 7, 0],
    list_next_days = [3, 3, 3],
    ):
    
    (
        list_images_3_days,
        list_images_7_days,
        list_images_30_days, 
        list_dates, 
        list_prices, 
        list_ema_9, 
        list_macd_histogram
    ) = function_generate(        
        df_date=df_date,
        df_original=df_original,
        list_previous_days=list_previous_days,
        list_next_days=list_next_days,
        days_result=days_result,
        figscale=figscale,
        indicator=indicator,
    )
    
            
    list_images_3_days = np.array(list_images_3_days)
    list_images_7_days = np.array(list_images_7_days)
    list_images_30_days = np.array(list_images_30_days)
    list_prices = np.array(list_prices)
    list_dates = np.array(list_dates)
    list_ema_9 = np.array(list_ema_9)
    list_macd_histogram = np.array(list_macd_histogram)
    

    print(f"list_images_3_days shape = {list_images_3_days.shape}")
    print(f"list_images_7_days shape = {list_images_7_days.shape}")
    print(f"list_images_30_days shape = {list_images_30_days.shape}")
    print(f"list_prices shape = {list_prices.shape}")
    print(f"list_dates shape = {list_dates.shape}")
    print(f"list_ema_9 shape = {list_ema_9.shape}")
    print(f"list_macd_histogram shape = {list_macd_histogram.shape}")
    
    if len(list_images_3_days) == 0:
        return None
    
    dataset = tf.data.Dataset.from_tensor_slices((
        list_images_30_days,
        list_images_7_days,
        list_images_3_days,
        list_ema_9,
        list_macd_histogram,
        list_prices,
        list_dates,
    ))
    
    if save_dataset_to_folder is not None:
        if not os.path.exists(save_dataset_to_folder):
            os.makedirs(save_dataset_to_folder)
        else:
            exist_ok = input(f"Folder {save_dataset_to_folder} already exists, continue? [y/n]: ").lower()[0] == "y"
            if not exist_ok:
                return dataset
    
        tf.data.Dataset.save(dataset, save_dataset_to_folder)
    
    return dataset
