import keras
from functools import partial
from utils.common_train_utils import *
import pandas as pd
import numpy as np
import tensorflow as tf


def evaluate_btc_model_5(candle_type_and_directory_save, batch_size, shuffle_buffer):
    dataset_folder = "../dataset/btc_dataset_0_2_3_2014_2024_test_2023_2024"
    dataset_test = load_dataset_of_each_type_and_combine(f"{dataset_folder}/test", candle_type_and_directory_save)
    dataset_test_1 = dataset_test.shuffle(shuffle_buffer)
    dataset_test_2 = dataset_test_1.batch(batch_size)
    transform_function = partial(get_open_close_prices_percent_of_last_days_result, 3)
    dataset_test_3 = dataset_test_2.map(transform_function)
    
    model = keras.models.load_model(f"{dataset_folder}/model_save/model_5.keras")
    return model, model.evaluate(dataset_test_3, verbose=2)


def evaluate_fpt_model_5(candle_type_and_directory_save, batch_size, shuffle_buffer):
    dataset_folder = "../dataset/fpt_dataset_0_2_3_2006_2020_test_2019_2020"
    dataset_test = load_dataset_of_each_type_and_combine(f"{dataset_folder}/test", candle_type_and_directory_save)
    dataset_test_1 = dataset_test.shuffle(shuffle_buffer)
    dataset_test_2 = dataset_test_1.batch(batch_size)
    transform_function = partial(get_open_close_prices_percent_of_last_days_result, 3)
    dataset_test_3 = dataset_test_2.map(transform_function)
    
    model = keras.models.load_model(f"{dataset_folder}/model_save/model_5.keras")
    return model, model.evaluate(dataset_test_3, verbose=2)


def evaluate_fpt_model_5_biLSTM_with_trend_type(candle_type_and_directory_save, batch_size, shuffle_buffer):
    dataset_folder = "../dataset/fpt_dataset_0_2_3_2006_2020_test_2019_2020_with_trend_type"
    dataset_test = load_dataset_of_each_type_and_combine(f"{dataset_folder}/test", candle_type_and_directory_save)
    dataset_test_1 = dataset_test.shuffle(shuffle_buffer)
    dataset_test_2 = dataset_test_1.batch(batch_size)
    transform_function = partial(get_open_close_prices_percent_of_last_days_result_for_trend_type_dataset, 3)
    dataset_test_3 = dataset_test_2.map(transform_function)
    
    model = keras.models.load_model(f"{dataset_folder}/model_save/model_5_biLSTM_with_trend_type/model_5_biLSTM_with_trend_type.keras")
    return model, model.evaluate(dataset_test_3, verbose=2)


def evaluate_fpt_model_5_biLSTM_with_trend_type_2023_2024(candle_type_and_directory_save, batch_size, shuffle_buffer):
    dataset_folder = "../dataset/fpt_dataset_0_2_3_2012_2024_test_2023_2024_with_ema_macd_trend"
    dataset_test = load_dataset_of_each_type_and_combine(f"{dataset_folder}/test", candle_type_and_directory_save)
    dataset_test_1 = dataset_test.shuffle(shuffle_buffer)
    dataset_test_2 = dataset_test_1.batch(batch_size)
    transform_function = partial(get_open_close_prices_percent_of_last_days_result_for_ema_macd_trend_dataset, 3)
    dataset_test_3 = dataset_test_2.map(transform_function)
    model_name = "model_5_biLSTM_with_ema_macd_trend_1_1"
    model = keras.models.load_model(f"{dataset_folder}/model_save/{model_name}/{model_name}.keras")
    return model, model.evaluate(dataset_test_3)


def get_origin_transform_function(config_file_path):
    with open(config_file_path, 'r') as file:
        config_and_loss = json.load(file)
        model_function_name = config_and_loss["config"]["model_function"]
        model_function = globals().get(model_function_name)
        return model_function


def evaluate_one_model(model_path, config_file_path, dataset_test, days_result):
    model_loaded: keras.Sequential = keras.models.load_model(model_path)
    
    transform_function = get_origin_transform_function(config_file_path)
    dataset_test_transformed = dataset_test.map(
        partial(transform_function, days_result)
    )
    
    loss = model_loaded.evaluate(dataset_test_transformed)
    return loss


def evaluate_all_models_in_dataset_folder(all_models_folder_path, dataset_test, days_result):
    all_models_path_and_name = [
        (
            os.path.join(
                os.path.abspath(f.path), f"{f.name}.keras"
            ), 
            f.name,
            os.path.join(
                os.path.abspath(f.path), "config_and_loss.json"
            )
        ) 
        for f in os.scandir(all_models_folder_path) 
        if f.is_dir()
    ]
    
    list_models_name = []
    list_loss_mae = []
    list_loss_mse = []
    for model_path, model_name, config_file_path in all_models_path_and_name:
        loss = evaluate_one_model(model_path, config_file_path, dataset_test, days_result)
        list_models_name.append(model_name)
        list_loss_mse.append(loss[0])
        list_loss_mae.append(loss[1])
    
    df_result = pd.DataFrame({
        "model_name": list_models_name,
        "mse": list_loss_mse,
        "mae": list_loss_mae
    })
    
    df_result.sort_values("mse", inplace=True)
    return df_result
