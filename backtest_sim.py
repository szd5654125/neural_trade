from utils.data_processing import (load_csv_data, log_transform_column, add_time_features, calculate_percentage_change,
                                   calculate_average, process_columns, save_result_to_file, drop_rows_with_none, add_target_column)
from utils.indicator import calc_up_and_down, calculate_ema
from tensorflow.keras.models import load_model
import pandas as pd
import os

'''{"columns": ["average_df", "quote_volume"], "function": calc_up_and_down, "kwargs": {"dev": 2.72, 'length': 1130},
 "task_name": "up_and_down"},
{"columns": ["open", "up_and_down"], "function": calculate_percentage_change},'''

processing_tasks = [
    {"columns": "open_time", "function": add_time_features},
    {"columns": "open", "function": log_transform_column},
    {"columns": ["open", 'high'], "function": calculate_percentage_change},
    {"columns": ["open", 'low'], "function": calculate_percentage_change},
    {"columns": ["open", 'close'], "function": calculate_percentage_change},
    {"columns": "quote_volume", "function": log_transform_column},
    {"columns": ["low", "high"], "function": calculate_average, "task_name": "average_df"},
    {"columns": "close", "function": calculate_ema, "kwargs": {"period": 20}, "task_name": "ema10"},
    {"columns": ["open", "ema10"], "function": calculate_percentage_change},
    {"columns": "close", "function": calculate_ema, "kwargs": {"period": 80}, "task_name": "ema10"},
    {"columns": ["open", "ema10"], "function": calculate_percentage_change},
    {"columns": "close", "function": calculate_ema, "kwargs": {"period": 480}, "task_name": "ema10"},
    {"columns": ["open", "ema10"], "function": calculate_percentage_change},
    {"columns": "close", "function": calculate_ema, "kwargs": {"period": 960}, "task_name": "ema10"},
    {"columns": ["open", "ema10"], "function": calculate_percentage_change},
    {"columns": "close", "function": calculate_ema, "kwargs": {"period": 1440}, "task_name": "ema10"},
    {"columns": ["open", "ema10"], "function": calculate_percentage_change}
]

current_folder = os.getcwd()
file_path = os.path.join(current_folder, "data", "futures_um_monthly_klines_BTCUSDT_15m_0_55.csv")
df = pd.read_csv(file_path)
result = process_columns(df, processing_tasks)
result = drop_rows_with_none(result)
# save_result_to_file(result)
model = load_model("saved_model/mlp.keras")
predictions = model.predict(result)
result['predictions'] = predictions
save_result_to_file(result)