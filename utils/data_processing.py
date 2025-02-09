import pandas as pd
import os
import numpy as np
from datetime import datetime


def load_csv_data(file_name, folder="data"):
    # 构造完整路径
    file_path = os.path.join(folder, file_name)
    # 读取 CSV 文件
    try:
        df = pd.read_csv(file_path, header=0)  # 第一行是列名
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {file_path} 不存在，请检查路径或文件名。")
    return df


def calculate_average(df1, df2):
    # 检查输入是否为 DataFrame
    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        raise TypeError("输入必须是 Pandas DataFrame。")
    # 检查两者是否有相同的列数和索引
    if df1.shape[1] != df2.shape[1]:
        raise ValueError("两个 DataFrame 必须具有相同的列数。")
    if not df1.index.equals(df2.index):
        raise ValueError("两个 DataFrame 必须具有相同的索引。")
    # 初始化结果 DataFrame
    average_df = pd.DataFrame(index=df1.index)
    # 遍历两者的每一列
    for col1, col2 in zip(df1.columns, df2.columns):
        # 计算平均值
        average_series = (df1[col1] + df2[col2]) / 2
        # 生成新列名
        new_column_name = f"{col1}_{col2}_average"
        # 添加到结果 DataFrame
        average_df[new_column_name] = average_series
    return average_df


def log_transform_column(dataframe, epsilon=1e-9):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("输入必须是一个 DataFrame")
    # 检查 DataFrame 是否只有一列
    if dataframe.shape[1] != 1:
        raise ValueError("输入的 DataFrame 必须只有一列")
    # 获取列名
    original_column_name = dataframe.columns[0]
    # 进行对数变换
    transformed_column = np.log(dataframe[original_column_name] + epsilon)
    # 创建新的 DataFrame，列名添加 '_log' 后缀
    new_column_name = f"{original_column_name}_log"
    transformed_dataframe = pd.DataFrame({new_column_name: transformed_column})
    return transformed_dataframe


def add_time_features(df, time_column='open_time'):
    # 确保时间列为 datetime 格式
    df[time_column] = pd.to_datetime(df[time_column])
    # 提取时间相关特征
    df['year'] = df[time_column].dt.year
    df['month'] = df[time_column].dt.month
    df['day'] = df[time_column].dt.day
    df['hour'] = df[time_column].dt.hour
    df['minute'] = df[time_column].dt.minute
    # 计算周期性特征的正弦和余弦
    df['year_cycle'] = df['year'] % 4
    df['year_sin'] = np.sin(2 * np.pi * df['year_cycle'] / 4)
    df['year_cos'] = np.cos(2 * np.pi * df['year_cycle'] / 4)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)  # 假设每月 31 天
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    # 筛选正弦和余弦特征列
    sin_cos_columns = [col for col in df.columns if col.endswith('_sin') or col.endswith('_cos')]
    # 返回仅包含正弦和余弦特征的 DataFrame
    return df[sin_cos_columns]


#  将输入的其他价格对open做一个相对的百分比变化
def calculate_percentage_change(open_prices, *other_prices):
    """
    计算多个 other_prices 每列相对于 open_prices 的百分比变化。

    参数:
    - open_prices: pd.Series 或单列 pd.DataFrame，第一个输入，单列的价格数据。
    - *other_prices: 可变数量的 pd.DataFrame 或 pd.Series，第二个输入及之后的输入，多列的价格数据。

    返回:
    - pd.DataFrame，包含计算的百分比变化结果，每列以 "{列名}_percentage" 命名。
    """
    if isinstance(open_prices, pd.DataFrame) and open_prices.shape[1] != 1:
        raise ValueError("open_prices 必须是单列的 DataFrame。")
    # 如果 open_prices 是 DataFrame，提取为 Series
    if isinstance(open_prices, pd.DataFrame):
        open_prices = open_prices.squeeze()

    # 初始化结果 DataFrame
    percentage_changes = pd.DataFrame(index=open_prices.index)

    # 遍历所有的 other_prices
    for idx, other_df in enumerate(other_prices):
        # 检查 other_df 是否为 DataFrame 或 Series
        if isinstance(other_df, pd.Series):
            other_df = other_df.to_frame(name=f"other_{idx}")
        elif not isinstance(other_df, pd.DataFrame):
            raise TypeError(f"other_prices[{idx}] 必须是 Pandas DataFrame 或 Series。")
        # 检查索引是否一致
        if not open_prices.index.equals(other_df.index):
            raise ValueError(f"open_prices 和 other_prices[{idx}] 必须具有相同的索引。")
        # 遍历 other_df 的每一列，计算百分比变化
        for col in other_df.columns:
            percentage_changes[f"{col}_percentage"] = ((other_df[col] - open_prices) / open_prices) * 100

    return percentage_changes


def process_columns(df, processing_tasks):
    """
    根据指定的任务列表对 DataFrame 的列或列组合进行处理。
    支持多列同时作为输入。

    参数:
    - df: pd.DataFrame，输入的数据框。
    - processing_tasks: list，每个任务是一个字典，定义如下：
        [
            {
                "columns": ["open", "high"],  # 列名或列组合
                "function": calculate_average,  # 单一处理函数
                "kwargs": {"param1": value},  # 可选参数
                "task_name": "average_result"  # 可选任务名
            }
        ]

    返回:
    - dict，包含每个任务的结果，键为任务名（task_name）。
    """
    result_cache = {}  # 缓存每一步的结果
    result_df = pd.DataFrame()  # 用于存储每次计算后的数据
    for task in processing_tasks:
        columns = task["columns"]
        func = task.get("function")  # 处理函数
        # print(func)
        kwargs = task.get("kwargs", {})  # 函数参数
        task_name = task.get("task_name")  # 任务名

        if isinstance(columns, list):
            # 多列时逐列检查
            data_to_process = pd.DataFrame()
            for col in columns:
                if col in result_cache:
                    # 检查缓存中的数据是否是 DataFrame
                    if isinstance(result_cache[col], pd.DataFrame):
                        # 将多列结果扩展为多列
                        for sub_col in result_cache[col].columns:
                            data_to_process[f"{col}_{sub_col}"] = result_cache[col][sub_col].copy()
                    else:
                        # 单列直接赋值
                        data_to_process[col] = result_cache[col].copy()
                elif col in df.columns:
                    data_to_process[col] = df[col].copy()
                else:
                    raise ValueError(f"Invalid column: {col}")
        elif isinstance(columns, str):
            # 单列处理
            if columns in result_cache:
                data_to_process = result_cache[columns].copy()
            elif columns in df.columns:
                data_to_process = df[[columns]].copy()
            else:
                raise ValueError(f"Invalid column: {columns}")
        else:
            raise ValueError(f"Invalid columns: {columns}")

        if isinstance(data_to_process, pd.DataFrame) and len(data_to_process.columns) > 1:
            # 将第一列提取为单独的 DataFrame
            col1 = data_to_process.columns[0]
            df1 = data_to_process[[col1]].copy()

            # 将其余所有列合并为一个 DataFrame
            remaining_cols = data_to_process.columns[1:]
            df2 = data_to_process[remaining_cols].copy()

            # 调用函数，传入 df1 和 df2
            data_to_process = func(df1, df2, **kwargs)
        else:
            # print(data_to_process.shape)
            # 默认直接处理
            data_to_process = func(data_to_process, **kwargs)

        # 将当前任务的结果存入缓存
        if task_name:
            result_cache[task_name] = data_to_process
            # 如果 data_to_process 是 DataFrame，将其合并到结果
        if isinstance(data_to_process, pd.DataFrame):
            if task_name:
                # 添加前缀以区分任务结果列
                data_to_process.columns = [f"{task_name}_{col}" for col in data_to_process.columns]
            else:
                result_df = pd.concat([result_df, data_to_process], axis=1)
    return result_df


def save_result_to_file(result, folder_name="result", file_extension="csv"):
    """
    保存结果到指定文件夹，并以当前时间作为文件名。

    参数:
    - result: pd.DataFrame，要保存的 DataFrame。
    - folder_name: str，保存的文件夹名（默认为 "result"）。
    - file_extension: str，文件扩展名（默认为 "csv"）。

    返回:
    - 保存的文件路径。
    """
    # 获取当前文件夹路径
    current_folder = os.getcwd()
    # 构建目标文件夹路径
    result_folder = os.path.join(current_folder, folder_name)

    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # 生成文件名，以当前时间命名
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")  # 格式为 YYYYMMDDHHMMSS
    file_name = f"{current_time}.{file_extension}"

    # 构建完整的文件路径
    file_path = os.path.join(result_folder, file_name)

    # 保存文件
    if file_extension == "csv":
        result.to_csv(file_path, index=False)  # 保存为 CSV 文件，不保存索引
    elif file_extension == "xlsx":
        result.to_excel(file_path, index=False)  # 保存为 Excel 文件
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    print(f"File saved to: {file_path}")
    return file_path


def add_target_column(df, column_name):
    if column_name not in df.columns:
        raise ValueError(f"列 '{column_name}' 不在 DataFrame 中。")
    # 创建新列名
    target_column_name = f"{column_name}_target"
    # 复制列并平移
    df[target_column_name] = df[column_name].shift(-1)
    # 删除平移后超出索引范围的行
    df = df[:-1]
    return df


def drop_rows_with_none(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是 Pandas DataFrame 类型。")
    # 使用 dropna 方法删除包含 None 或 NaN 的行
    cleaned_df = df.dropna(how='any')
    return cleaned_df
