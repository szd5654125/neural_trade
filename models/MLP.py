from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, PReLU, Dropout
from tensorflow.keras.regularizers import l1, l2, l1_l2


def build_mlp_model(input_dim, output_type='regression', num_classes=None, dropout_rate=0.2, l1_lambda=0.0, l2_lambda=0.01):
    if output_type not in ['regression', 'classification']:
        raise ValueError("output_type 必须是 'regression' 或 'classification'")
    if output_type == 'classification' and num_classes is None:
        raise ValueError("分类任务必须提供 num_classes 参数")
    # 正则化配置
    regularizer = l1_l2(l1=l1_lambda, l2=l2_lambda)
    # 创建模型
    model = Sequential()
    # 输入层 + 第一隐藏层
    model.add(Input(shape=(input_dim,)))  # 输入层
    model.add(Dense(64, kernel_regularizer=regularizer))  # 第一隐藏层（32个神经元）
    model.add(PReLU())  # 激活函数：Parametric ReLU
    model.add(Dropout(dropout_rate))
    # 第二隐藏层
    model.add(Dense(128, kernel_regularizer=regularizer))  # 第二隐藏层（16个神经元）
    model.add(PReLU())  # 激活函数：Parametric ReLU
    model.add(Dropout(dropout_rate))
    # 新增第三隐藏层
    '''model.add(Dense(32, kernel_regularizer=regularizer))  # 第三隐藏层（32个神经元）
    model.add(PReLU())  # 激活函数：Parametric ReLU
    model.add(Dropout(dropout_rate))
    # 新增第四隐藏层
    model.add(Dense(16, kernel_regularizer=regularizer))  # 第四隐藏层（16个神经元）
    model.add(PReLU())  # 激活函数：Parametric ReLU
    model.add(Dropout(dropout_rate))'''
    # 输出层
    if output_type == 'regression':
        model.add(Dense(1, activation='linear'))  # 回归任务：线性激活
    elif output_type == 'classification':
        model.add(Dense(num_classes, activation='softmax'))  # 分类任务：softmax 激活

    # 编译模型
    if output_type == 'regression':
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # 回归使用均方误差
    elif output_type == 'classification':
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # 分类使用交叉熵

    return model