import json
import tensorflow as tf
import keras
import numpy as np


def read_config(dataset_folder):
    config = None
    with open(f"{dataset_folder}/config.json", "r") as f:
        config = json.load(f)
        return config
    

def load_dataset_test(dataset_folder):
    return tf.data.Dataset.load(f'{dataset_folder}/test')


def load_dataset_of_each_type_and_combine(dataset_folder, candle_type_and_directory_save: dict):    
    list_dataset = []
    for folder_name in candle_type_and_directory_save.values():
        try:
            dataset = tf.data.Dataset.load(f'{dataset_folder}/{folder_name}')
            print(f"folder: {folder_name}, \t\t total: {len(dataset)} images")
            list_dataset.append(dataset)
            
        except tf.errors.NotFoundError:
            print(f"folder: {folder_name}, \t\t total: 0 images")
            continue
    dataset_train = list_dataset[0]
    for dataset in list_dataset[1:]:
        dataset_train = dataset_train.concatenate(dataset)
    
    return dataset_train


def get_close_price_percent_of_last_days_result(days_result, features, labels, *args, **kwargs):
    last_close_in_candle = labels[:,-days_result-1:-days_result, 2]
    closes = labels[:,-days_result:,2]
    closes = ((closes - last_close_in_candle) / last_close_in_candle) * 100
    return features, closes


def get_prices_percent_of_last_days_result(days_result, features, labels, *args, **kwargs):
    prices_of_last_candle_in_image = labels[:,-days_result-1:-days_result]
    prices_of_last_days_result = labels[:,-days_result:]
    prices_percent = ((prices_of_last_days_result - prices_of_last_candle_in_image) / prices_of_last_candle_in_image) * 100
    return features, prices_percent


def get_high_low_prices_percent_of_last_days_result(days_result, features, labels, *args, **kwargs):
    prices_of_last_candle_in_image = tf.gather(labels[:,-days_result-1:-days_result, :], [0, 3], axis=-1)
    prices_of_last_days_result = tf.gather(labels[:, -days_result:, :], [0, 3], axis=-1)
    prices_percent = ((prices_of_last_days_result - prices_of_last_candle_in_image) / prices_of_last_candle_in_image) * 100
    prices_percent = prices_percent[:,-1]
    return features, prices_percent


def swap_max_and_min_two_columns(inputs, column_1, column_2):
    # Chọn các giá trị ở index 1 và 2 của mỗi timestep trong feature
    selected_features = tf.gather(inputs, [column_1, column_2], axis=-1)  # Shape: (batch_size, timestep, 2)
    
    # Tìm giá trị max và min ở cặp (1, 2)
    max_vals = tf.reduce_max(selected_features, axis=-1, keepdims=True)  # Shape: (batch_size, timestep, 1)
    min_vals = tf.reduce_min(selected_features, axis=-1, keepdims=True)  # Shape: (batch_size, timestep, 1)

    # Kết hợp max và min thành tensor mới có max tại index 1 và min tại index 2
    swapped_features = tf.concat([max_vals, min_vals], axis=-1)  # Shape: (batch_size, timestep, 2)

    # Ghép các feature lại với nhau, thay thế các giá trị ở index 1 và 2 bằng tensor đã hoán đổi
    outputs = tf.concat([inputs[:, :, :column_1], swapped_features, inputs[:, :, column_2:]], axis=-1)

    return outputs


def get_open_close_prices_percent_of_last_days_result_for_one_day_result(days_result, features, labels, *args, **kwargs):
    labels = swap_max_and_min_two_columns(labels, 1, 2)
    open_close_prices_of_last_candle_in_image = tf.gather(labels[:,-days_result-1:-days_result, :], [1, 2], axis=-1)
    open_close_prices_of_last_days_result = tf.gather(labels[:, -days_result:, :], [1, 2], axis=-1)
    prices_percent = ((open_close_prices_of_last_days_result - open_close_prices_of_last_candle_in_image) / open_close_prices_of_last_candle_in_image) * 100
    prices_percent = prices_percent[:,-1]
    return features, prices_percent


def get_open_close_prices_percent_of_last_days_result_for_multiple_days_result(days_result, features, labels, *args, **kwargs):
    labels = swap_max_and_min_two_columns(labels, 1, 2)
    
    difference = labels[:, 1:] - labels[:, :-1]
    percent_change = (difference / labels[:, :-1]) * 100
    
    percent_change_of_open_close = tf.gather(percent_change[:, -days_result:, :], [1, 2], axis=-1 )
    return features, percent_change_of_open_close


def get_open_close_prices_percent_of_last_days_result(days_result, features, labels, *args, **kwargs):
    if days_result > 1:
        return get_open_close_prices_percent_of_last_days_result_for_multiple_days_result(days_result, features, labels, *args, **kwargs)
    return get_open_close_prices_percent_of_last_days_result_for_one_day_result(days_result, features, labels, *args, **kwargs)


def get_open_close_prices_percent_of_last_days_result_for_trend_type_dataset(days_result, trend_type_values, features, labels, *args, **kwargs):
    features, percent_change_of_open_close = get_open_close_prices_percent_of_last_days_result(days_result, features, labels, *args, **kwargs)
    return ((trend_type_values, features), percent_change_of_open_close)


def get_open_close_log_diff_of_last_days_result_for_one_day_result(days_result, features, labels, *args, **kwargs):
    labels = swap_max_and_min_two_columns(labels, 1, 2)
    open_close_prices_of_last_candle_in_image = tf.gather(labels[:,-days_result-1:-days_result, :], [1, 2], axis=-1)
    open_close_prices_of_last_days_result = tf.gather(labels[:, -days_result:, :], [1, 2], axis=-1)
    prices_percent = ((open_close_prices_of_last_days_result - open_close_prices_of_last_candle_in_image) / open_close_prices_of_last_candle_in_image) * 100
    prices_percent = prices_percent[:,-1]
    return features, prices_percent


def get_open_close_log_diff_of_last_days_result_for_multiple_days_result(days_result, features, labels, *args, **kwargs):
    labels = swap_max_and_min_two_columns(labels, 1, 2)
    
    difference = labels[:, 1:] / labels[:, :-1]
    log_change = np.log(difference)
    
    log_change_of_open_close = tf.gather(log_change[:, -days_result:, :], [1, 2], axis=-1 )
    return features, log_change_of_open_close


def get_open_close_log_diff_of_last_days_result(days_result, features, labels, *args, **kwargs):
    if days_result > 1:
        return get_open_close_prices_percent_of_last_days_result_for_multiple_days_result(days_result, features, labels, *args, **kwargs)
    return get_open_close_prices_percent_of_last_days_result_for_one_day_result(days_result, features, labels, *args, **kwargs)


def get_max_in_open_close_prices_percent_of_last_days_result_for_multiple_days_result(days_result, features, labels, *args, **kwargs):
    labels = swap_max_and_min_two_columns(labels, 1, 2)
    
    difference = labels[:, 1:] - labels[:, :-1]
    percent_change = (difference / labels[:, :-1]) * 100
    
    percent_change_of_open_close = tf.gather(percent_change[:, -days_result:, :], [1], axis=-1 )
    return features, percent_change_of_open_close


def up_or_down_in_close_for_multiple_days_result(days_result, features, labels, *args, **kwargs):
    difference = labels[:, 1:] - labels[:, :-1]

    binary_difference = tf.where(difference >= 0, 1, 0)
    
    up_or_down = tf.gather(binary_difference[:, -days_result:, :], [2], axis=-1 )
    one_hot_labels = tf.one_hot(tf.squeeze(up_or_down, axis=-1), depth=2)
    return features, one_hot_labels


def up_or_down_in_close_for_one_days_result(days_result, features, labels, *args, **kwargs):
    difference = labels[:, 1:] - labels[:, :-1]

    binary_difference = tf.where(difference >= 0, 1, 0)
    
    up_or_down = tf.gather(binary_difference[:, -days_result:-days_result+1, :], [2], axis=-1 )
    one_hot_labels = tf.one_hot(tf.squeeze(up_or_down, axis=-1), depth=2)
    
    return features, one_hot_labels


def get_image_shape(dataset):
    pass
