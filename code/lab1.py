import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 数据读入
data = pd.read_csv("boston.csv")
# print(data)

# 定义特征值
features = data[['crim', 'rm', 'lstat']]

for ii in features.columns:
    plt.figure()
    plt.title(ii)
    plt.hist(features[ii], bins=50, color='steelblue', density=True)
    plt.show()

# 定义目标值
target = data['medv']
# 输出特征值描述性统计

# 区分训练集和测试集
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=10)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 线性回归
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
print(lr.coef_, lr.intercept_)

# 预测
y_pred = lr.predict(x_test)

x_min = min(np.min(y_test), np.min(y_pred))
x_max = max(np.max(y_test), np.max(y_pred))
plt.scatter(y_test, y_pred)
plt.plot([x_min, x_max], [x_min, x_max], c='g')
plt.show()

# 获取平均绝对误差MAE
from sklearn.metrics import mean_squared_error, mean_absolute_error

mae_test = mean_absolute_error(y_test, y_pred)

# 获取均分误差MSE
mse_test = mean_squared_error(y_test, y_pred)
print("MSE:{}\nMAE:{}".format(mse_test, mae_test))
