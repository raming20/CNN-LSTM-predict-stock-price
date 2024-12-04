import sys
sys.path.append("../")

from datetime import datetime
from functools import partial
from matplotlib.dates import DateFormatter
from ta import add_all_ta_features
from ta.utils import dropna
from utils.common_train_utils import *
from utils.draw_candle_image import *
from utils.evaluate_old_models import *
import json
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import shutil
import ta
import ta.momentum
import ta.trend
import tempfile
import tensorflow as tf
import matplotlib.dates as mdates
keras.config.enable_unsafe_deserialization()
from utils.draw_candle_image import draw_candle_image
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import imagehash



def check_last_candle_is_up_or_down(df: pd.DataFrame, day_result):
    open = df.iloc[-day_result, 1]
    close = df.iloc[-day_result, 2]
    
    if open <= close:
        return "up"
    
    return "down"


def calculate_prediction_df(
    df_real_data: pd.DataFrame,
    predictions_i: np.ndarray,
    days_result: int,
    extend_real: bool,
    is_trading_24h=False,
    ):
    # ["High", "Open", "Close", "Low"]
    # [0, 1, 2, 3]
    
    df_predictions = df_real_data.copy()
    df_real_data_extend = df_real_data.copy()

    for day in range(days_result, 0, -1):
        is_real_up = check_last_candle_is_up_or_down(df_real_data, day) == "up"
        open_close_real_before = df_predictions.iloc[-day-1, [1, 2]]
        max_in_open_close_real_before = np.max(open_close_real_before)
        min_in_open_close_real_before = np.min(open_close_real_before)
        
        if days_result == 1:
            percent_predict = predictions_i
        else:
            percent_predict = predictions_i[-day]
        
        swaped = [max_in_open_close_real_before, min_in_open_close_real_before]
        result = (percent_predict * swaped) / 100 + swaped
        
        max_in_result = np.max(result)
        min_in_result = np.min(result)
        
        if is_real_up:
            df_predictions.iloc[-day, [1, 2]] = [min_in_result, max_in_result]
            if extend_real: df_real_data_extend.iloc[-day, [1, 2]] = df_real_data_extend.iloc[-day, [3, 0]]
        else:
            df_predictions.iloc[-day, [1, 2]] = [max_in_result, min_in_result]
            if extend_real: df_real_data_extend.iloc[-day, [1, 2]] = df_real_data_extend.iloc[-day, [0, 3]]
        df_predictions.iloc[-day, [0, 3]] = [max_in_result, min_in_result]
    
    if is_trading_24h:
        for day in range(days_result, 0, -1):
            is_real_up = check_last_candle_is_up_or_down(df_real_data, day) == "up"
            close_price_before = df_predictions.iloc[-day-1, [2]]
            df_predictions.iloc[-day, [1]] = close_price_before
            
    return df_predictions, df_real_data_extend
    


def draw_prediction(
    x_dataset_test_1_i, 
    y_dataset_test_1_i, 
    date_i,
    predictions_i, 
    days_result=3,
    show_x_orginal_candle=False, 
    show_original_candle=False,
    show_prediction_candle=False,
    show_close_compare=False,
    show_open_compare=False,
    type_of_output=None,
    draw_beside=False,
    save_image=None,
    print_image=True,
    extend_real=False,
    date_generator=lambda x: x,
    symbol="None",
    print_all_days_for_prediction=True,
    days_prediction_to_print=None,
    is_trading_24h=False
    ):
    
    
    all_dates, total_date_reals = date_generator(date_i)
    year = int(all_dates[0].year)
    
    days_predict_for = days_result
    if not print_all_days_for_prediction:
        days_predict_for = total_date_reals - (len(all_dates) - days_result)
    
    if days_prediction_to_print is not None:
        days_predict_for = days_prediction_to_print
        total_date_reals = (len(all_dates) - days_result) + days_prediction_to_print
    
    symbol_and_year_str_title = f"mã {symbol}, năm {year}"
    title_input = f"""Biểu đồ giá đầu vào ngày {date_i}: \n{symbol_and_year_str_title}"""
    title_real_candle = f"""Biểu đồ giá cổ phiếu thực tế: \n{symbol_and_year_str_title}"""
    title_predict_candle = f"""Biểu đồ giá cổ phiếu dự đoán cho {days_predict_for} ngày cuối của đồ thị: \n{symbol_and_year_str_title}"""
    title_close = f"""Biểu đồ giá đóng cửa: \n{symbol_and_year_str_title}"""
    title_open = f"""Biểu đồ giá mở cửa: \n{symbol_and_year_str_title}"""
    
    label_real_close='Giá đóng cửa thật'
    label_predict_close=f'Giá đóng cửa dự đoán cho {days_predict_for} ngày cuối'
    label_real_open='Giá mở cửa thật'
    label_predict_open=f'Giá mở cửa dự đoán cho {days_predict_for} ngày cuối'
    
    if show_x_orginal_candle and not draw_beside:
        plt.imshow(x_dataset_test_1_i)
        if save_image is not None:
            plt.savefig(save_image)
    
    df_real_data = pd.DataFrame(y_dataset_test_1_i, columns=["High", "Open", "Close", "Low"])
    
    df_real_data["Date"] = all_dates
    df_real_data["Date"] = pd.to_datetime(df_real_data["Date"])
    df_real_data.set_index('Date', inplace=True)
    
    df_predictions, df_real_data_extend = calculate_prediction_df(
        df_real_data=df_real_data,
        predictions_i=predictions_i,
        days_result=days_result,
        extend_real=extend_real,
        is_trading_24h=is_trading_24h,
    )
    
    close_prices_real = df_real_data["Close"][0:total_date_reals]
    open_prices_real = df_real_data["Open"][0:total_date_reals]
    
    mc = mpf.make_marketcolors(
    up='green', down='red', wick='inherit', edge='inherit', volume='inherit', )
    style = mpf.make_mpf_style(marketcolors=mc, figcolor="white")
    
    if show_original_candle and not draw_beside:
        fig, axlist = mpf.plot(
            df_real_data[0:total_date_reals],
            type='candle',
            style=style,
            volume=False,  # Hiển thị khối lượng
            axisoff=False,  # Bỏ trục x và y
            returnfig=True,  # Trả về đối tượng Figure để tùy chỉnh
            figratio=(5, 5),  # Điều chỉnh tỷ lệ khung hình để thu hẹp khoảng cách
            figscale=1,  # Tăng kích thước biểu đồ để làm các nến gần nhau hơn
            panel_ratios=[6],
            title=title_real_candle,
        )
        # plt.tight_layout()
        if save_image is not None:
            plt.savefig(save_image, bbox_inches='tight')
    
    all_dates = df_real_data.index.strftime("%b %d")
    all_dates_of_prediction = all_dates
    if not print_all_days_for_prediction:
        all_dates_of_prediction = all_dates[0:total_date_reals]
        df_predictions = df_predictions[0:total_date_reals]
    
    if show_prediction_candle and not draw_beside:
        fig, axlist = mpf.plot(
            df_predictions,
            type='candle',
            style=style,
            volume=False,  # Hiển thị khối lượng
            axisoff=False,  # Bỏ trục x và y
            returnfig=True,  # Trả về đối tượng Figure để tùy chỉnh
            figratio=(5, 5),  # Điều chỉnh tỷ lệ khung hình để thu hẹp khoảng cách
            figscale=1,  # Tăng kích thước biểu đồ để làm các nến gần nhau hơn
            panel_ratios=[6],
            title=title_predict_candle,
        )
        if save_image is not None:
            plt.savefig(save_image, bbox_inches='tight')
            
    
    if show_close_compare and not draw_beside:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(all_dates[0:total_date_reals], close_prices_real, color='blue', marker='o', linestyle='-', label=label_real_close)
        plt.plot(all_dates_of_prediction, df_predictions["Close"], color='orange', marker='x', linestyle='--', label=label_predict_close)
        plt.title(title_close)
        plt.xlabel('Ngày')
        plt.ylabel('Giá Đóng cửa')
        plt.legend()
        plt.grid(True)
        plt.show()
        if save_image is not None:
            plt.savefig(save_image, bbox_inches='tight')
        
    if show_open_compare and not draw_beside:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(all_dates[0:total_date_reals], open_prices_real, color='blue', marker='o', linestyle='-', label=label_real_open)
        plt.plot(all_dates_of_prediction, df_predictions["Open"], color='orange', marker='x', linestyle='--', label=label_predict_open)
        plt.title(title_open)
        plt.xlabel('Ngày')
        plt.ylabel('Giá Đóng cửa')
        plt.legend()
        plt.grid(True)
        plt.show()
        if save_image is not None:
            plt.savefig(save_image, bbox_inches='tight')
        
    
    # Tạo Figure và hai Subplots cạnh nhau
    if draw_beside:
        fig, ((ax0, ax00), (ax1, ax2), (ax3, ax4)) = plt.subplots(3, 2, figsize=(12, 12))
        
        
        ax0.set_title(title_input)
        
        ax1.set_title(title_real_candle)
        ax2.set_title(title_predict_candle)
        
        ax3.set_title(title_close)
        ax4.set_title(title_open)
        
        if extend_real:
            ax00.set_title(f"""Biểu đồ khoảng giá cổ phiếu thực tế {days_predict_for} ngày cuối của đồ thị: \n{symbol_and_year_str_title}""")
            
            mpf.plot(
                df_real_data_extend[0:total_date_reals],
                type='candle',
                style=style,
                volume=False,  # Hiển thị khối lượng
                axisoff=False,  # Bỏ trục x và y
                ax=ax00
            )
            
        ax0.imshow(x_dataset_test_1_i)
        mpf.plot(
            df_real_data[0:total_date_reals],
            type='candle',
            style=style,
            volume=False,  # Hiển thị khối lượng
            axisoff=False,  # Bỏ trục x và y
            ax=ax1
        )
        mpf.plot(
            df_predictions,
            type='candle',
            style=style,
            volume=False,  # Hiển thị khối lượng
            axisoff=False,  # Bỏ trục x và y
            ax=ax2
        )
        
        ax3.plot(all_dates[0:total_date_reals], close_prices_real, color='blue', marker='o', linestyle='-', label=label_real_close)
        ax3.plot(all_dates_of_prediction, df_predictions["Close"], color='orange', marker='x', linestyle='--', label=label_predict_close)
        ax3.legend()
        ax3.grid(True)
        
        ax4.plot(all_dates[0:total_date_reals], open_prices_real, color='blue', marker='o', linestyle='-', label=label_real_open)
        ax4.plot(all_dates_of_prediction, df_predictions["Open"], color='orange', marker='x', linestyle='--', label=label_predict_open)

        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        if save_image is not None:
            plt.savefig(save_image)
        
    if print_image:
        plt.show()
    else:
        plt.close(fig)


def calculate_SSIM_between_two_image_tensor(image_tensor_1: np.ndarray, image_tensor_2: np.ndarray):
    image1 = np.dot(image_tensor_1[..., :3], [0.2989, 0.5870, 0.1140])  # Chuyển sang grayscale
    image2 = np.dot(image_tensor_2[..., :3], [0.2989, 0.5870, 0.1140])
    
    score, diff = ssim(image1, image2, full=True, data_range=image1.max() - image1.min())
    
    return score, diff


def calculate_phash_similarity(image_array1, image_array2):
    # Chuyển đổi numpy array sang định dạng PIL.Image
    image1 = Image.fromarray((image_array1 * 255).astype(np.uint8)) if image_array1.dtype == float else Image.fromarray(image_array1)
    image2 = Image.fromarray((image_array2 * 255).astype(np.uint8)) if image_array2.dtype == float else Image.fromarray(image_array2)

    # Tính toán pHash cho hai ảnh
    hash1 = imagehash.phash(image1)
    hash2 = imagehash.phash(image2)

    # Tính khoảng cách Hamming giữa hai hash
    hamming_distance = hash1 - hash2

    # Tính độ tương đồng
    similarity = 1 - hamming_distance / len(hash1.hash) ** 2

    return similarity, hamming_distance


def calculate_similarity_prediction_and_real(
    y_dataset_test_1_i, 
    predictions_i, 
    date_i,
    days_result=3,
    date_generator=lambda x: (x, 6),
    function_calculate_similarity=calculate_SSIM_between_two_image_tensor
    ):
    
    all_dates, total_date_reals = date_generator(date_i)
    year = int(all_dates[0].year)
    
    days_predict_for = total_date_reals - (len(all_dates) - days_result)
    
    df_real_data = pd.DataFrame(y_dataset_test_1_i, columns=["High", "Open", "Close", "Low"])
    
    df_real_data["Date"] = all_dates
    df_real_data["Date"] = pd.to_datetime(df_real_data["Date"])

    
    df_predictions, df_real_data_extend = calculate_prediction_df(
        df_real_data=df_real_data,
        predictions_i=predictions_i,
        days_result=days_result,
        extend_real=True
    )
    
    image_tensor_real = draw_candle_image(
        df_real_data[0:total_date_reals], 
        show_x_y=False, 
        show_volume=False, 
        
        figscale=0.5, 
        figcolor="black", 
        preview_image=False,
        return_image_tensor=True
    )
    
    image_tensor_extend_real = draw_candle_image(
        df_real_data_extend[0:total_date_reals], 
        show_x_y=False, 
        show_volume=False, 
        
        figscale=0.5, 
        figcolor="black", 
        preview_image=False,
        return_image_tensor=True
    )
    
    image_tensor_prediction = draw_candle_image(
        df_predictions[0:total_date_reals], 
        show_x_y=False, 
        show_volume=False, 
        
        figscale=0.5, 
        figcolor="black", 
        preview_image=False,
        return_image_tensor=True
    )
    
    
    score_real, diff_real = function_calculate_similarity(image_tensor_real, image_tensor_prediction)
    score_extend_real, diff_extend_real = function_calculate_similarity(image_tensor_extend_real, image_tensor_prediction)
    
    return {
        "date": date_i,
        "days_predict_for": days_predict_for,
        "score_real": score_real,
        "score_extend_real": score_extend_real,
        "diff_real": diff_real,
        "diff_extend_real": diff_extend_real,
    }


        
