import keras
from functools import partial
from utils.common_train_utils import get_open_close_prices_percent_of_last_days_result, load_dataset_of_each_type_and_combine


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


