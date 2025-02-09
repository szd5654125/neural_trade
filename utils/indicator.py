import pandas as pd
import numpy as np


def calculate_ema(dataframe, period):
    # print(dataframe,period)
    # 检查输入是否为 Pandas DataFrame
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe 必须是 Pandas DataFrame 类型。")
    # 检查 DataFrame 是否只有一列
    if dataframe.shape[1] != 1:
        raise ValueError("dataframe 必须只包含一列数据。")
    # 检查周期是否合法
    if period <= 0:
        raise ValueError("period 必须是正整数。")
    # 获取列名
    column_name = dataframe.columns[0]
    # 计算 EMA
    ema = dataframe[column_name].ewm(span=period, adjust=False).mean()
    ema[:period - 1] = None
    # 创建带列名的输出 DataFrame
    ema_df = pd.DataFrame({f"{column_name}_ema_{period}": ema.values}, index=dataframe.index)
    return ema_df


def calc_up_and_down(df_source, df_quote_volume, dev, length):
    np_source = np.asarray(df_source).flatten()  # 将dataframe转为series
    n = len(df_quote_volume)
    length_variable = np.full(n, np.nan)  # 初始化 NaN 数组
    for i in range(length - 1, n):
        window_subset = df_quote_volume[i - length + 1: i + 1]  # 选取窗口范围内的数据
        avg_value = np.mean(window_subset)  # 计算窗口内的平均值
        count_above_avg = np.sum(window_subset < avg_value)  # 计算大/小于平均值的个数
        length_variable[i] = count_above_avg
    # 计算完成后再进行替换
    length_variable[length_variable == 1] = 2
    n = len(np_source)
    up = np.full(n, np.nan)
    down = np.full(n, np.nan)
    # Compute intercepts and slopes using the custom function
    intercepts, slopes = compute_intercepts_and_slopes(np_source, length_variable)
    for i in range(n):
        l_i = int(length_variable[i]) if not np.isnan(length_variable[i]) else None
        if l_i is None or i < l_i - 1:
            continue  # Skip if invalid length or not enough data points
        # Compute lreg[i] and lreg_x[i] using intercepts and slopes
        lreg_i = intercepts[i] + slopes[i] * (l_i - 1)
        lreg_x_i = intercepts[i] + slopes[i] * (l_i - 2)
        s_i = lreg_i - lreg_x_i
        # Extract the window of source data
        window_start = i - l_i + 1
        window_end = i + 1
        window_subset = np_source[window_start:window_end]
        # Compute distances
        distances = np.arange(l_i - 1, -1, -1)
        # Process data
        processed_data = (window_subset + s_i * distances - lreg_i) ** 2
        ds = np.sum(processed_data)
        de = np.sqrt(ds / l_i)
        up[i] = (-de * dev) + np_source[i]
        down[i] = (de * dev) + np_source[i]
    # 将结果转换为 DataFrame
    result_df = pd.DataFrame({
        "up": up,
        "down": down
    }, index=df_source.index)
    return result_df


def compute_intercepts_and_slopes(y, length_variable):
    n = len(y)
    y = np.asarray(y)
    # Compute cumulative sums
    s_y = np.concatenate((np.array([0]), np.cumsum(y)))
    s_ty = np.concatenate((np.array([0]), np.cumsum(np.arange(n) * y)))

    # Prepare arrays to hold intercepts and slopes
    intercepts = np.full(n, np.nan)
    slopes = np.full(n, np.nan)

    # Precompute sum_x and sum_xx for possible window lengths
    unique_lengths = np.unique(length_variable[~np.isnan(length_variable)]).astype(int)
    sum_x_dict = {}
    sum_xx_dict = {}
    for length in unique_lengths:
        length = np.float64(length)  # Cast l to 64-bit float
        sum_x = (length - 1) * length / 2.0
        sum_xx = (length - 1) * length * (2 * length - 1) / 6.0
        sum_x_dict[length] = sum_x
        sum_xx_dict[length] = sum_xx

    for i in range(n):
        if np.isnan(length_variable[i]):
            intercepts[i] = np.nan
            slopes[i] = np.nan
            continue  # Skip to the next iteration
        l_i = int(length_variable[i])
        if i >= l_i - 1:
            t1 = i - l_i + 1
            t2 = i
            n_i = l_i

            sum_y = s_y[t2 + 1] - s_y[t1]
            sum_ty = s_ty[t2 + 1] - s_ty[t1]
            sum_x = sum_x_dict[n_i]
            sum_xx = sum_xx_dict[n_i]
            sum_xy = sum_ty - t1 * sum_y  # Adjusted sum_xy

            denominator = n_i * sum_xx - sum_x ** 2
            if denominator == 0:
                slope = 0
                intercept = sum_y / n_i
            else:
                slope = (n_i * sum_xy - sum_x * sum_y) / denominator
                intercept = (sum_y - slope * sum_x) / n_i

            intercepts[i] = intercept
            slopes[i] = slope
        else:
            intercepts[i] = np.nan
            slopes[i] = np.nan

    return intercepts, slopes
