from utils.data_processing import (load_csv_data, log_transform_column, add_time_features, calculate_percentage_change,
                                   calculate_average, process_columns, save_result_to_file, drop_rows_with_none, add_target_column)
from utils.indicator import calc_up_and_down, calculate_ema
from models.MLP import build_mlp_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
    {"columns": "close", "function": calculate_ema, "kwargs": {"period": 20}, "task_name": "ema1"},
    {"columns": ["open", "ema1"], "function": calculate_percentage_change},
    {"columns": "close", "function": calculate_ema, "kwargs": {"period": 80}, "task_name": "ema2"},
    {"columns": ["open", "ema2"], "function": calculate_percentage_change},
    {"columns": "close", "function": calculate_ema, "kwargs": {"period": 480}, "task_name": "ema3"},
    {"columns": ["open", "ema3"], "function": calculate_percentage_change},
    {"columns": "close", "function": calculate_ema, "kwargs": {"period": 960}, "task_name": "ema4"},
    {"columns": ["open", "ema4"], "function": calculate_percentage_change},
    {"columns": "close", "function": calculate_ema, "kwargs": {"period": 1440}, "task_name": "ema5"},
    {"columns": ["open", "ema5"], "function": calculate_percentage_change}
]

current_folder = os.getcwd()
file_path = os.path.join(current_folder, "data", "futures_um_monthly_klines_BTCUSDT_3m_0_7.csv")
df = pd.read_csv(file_path)
result = process_columns(df, processing_tasks)
result = add_target_column(result, 'close_percentage')
result = drop_rows_with_none(result)
# save_result_to_file(result)
# 分离出特征和标签
X = result.drop(columns=['close_percentage_target'])  # 假设 'target_column' 是你的标签列
y = result['close_percentage_target']
# 分离出训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = build_mlp_model(input_dim=X_train.shape[1], output_type='regression')
# 回调函数
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
history = model.fit(
    X_train, y_train,  # 训练数据
    validation_data=(X_test, y_test),  # 验证数据
    epochs=100,  # 训练轮数
    batch_size=512,  # 每批数据的大小
    verbose=1,  # 显示训练过程的日志
    callbacks=[early_stopping, reduce_lr]
)
loss, mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {loss}, Test MAE: {mae}")
model.save("saved_model/mlp.keras")

