from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge

def load_data():
    # 1.获取数据
    boston_data = load_boston()
    # print("特征数量为:(样本数,特征数)", boston_data.data.shape)
    # 2.数据处理-分开训练集和测试集    
    x_train, x_test, y_train, y_test = train_test_split(boston_data.data,
                                                        boston_data.target, 
                                                        random_state=1853799)
    return x_train, x_test, y_train, y_test


# 正规方程
def linear_Regression():
    
    # 3.特征工程
    # 3.1.承接数据集
    x_train, x_test, y_train, y_test = load_data()
    # 3.2.特征预处理
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    
    # 4.预估-预估器：线性回归-正规方程算法 
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    print("正规方程_系数为: ", estimator.coef_)
    print("正规方程_截距为:", estimator.intercept_)
    
    # 5.模型评估-均方差
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    # print("正规方程_房价预测:", y_predict)
    print("正规方程_均分误差:", error)
    return None


# 梯度下降
def linear_SGDRegressor():

    # 3.特征工程
    # 3.1.承接数据集
    x_train, x_test, y_train, y_test = load_data()
    # 3.2.特征预处理
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    
    # 4.预估-预估器：线性回归-梯度下降算法 
    estimator = SGDRegressor(learning_rate="constant", eta0=0.01, max_iter=10000)
    # 默认值如下
    # estimator = SGDRegressor(loss="squared_loss", fit_intercept=True, eta0=0.01,
    #                          power_t=0.25)
    # 如下设置就相当于岭回归, 但建议用Ridge方法
    # estimator = SGDRegressor(penalty='l2', loss="squared_loss")  
    estimator.fit(x_train, y_train)
    print("梯度下降_系数为: ", estimator.coef_)
    print("梯度下降_截距为:", estimator.intercept_)
    
    # 5.模型评估-均方差
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    # print("梯度下降_房价预测:", y_predict)
    print("梯度下降_均分误差:", error)

    return None

# 岭回归
def linear_Ridge():
    # 3.特征工程
    # 3.1.承接数据集
    x_train, x_test, y_train, y_test = load_data()
    transfer = StandardScaler()  # 标准化处理数据
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    
    # 4.预估-预估器：线性回归-岭回归算法 
    estimator = Ridge(max_iter=10000, alpha=0.5)
    # estimator = RidgeCV(alphas=[0.1, 0.2, 0.3, 0.5])  # 加了交叉验证的岭回归
    estimator.fit(x_train, y_train)
    print("岭回归_系数为: ", estimator.coef_)
    print("岭回归_截距为:", estimator.intercept_)
    
    # 5.模型评估-均方差
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
   #  print("岭回归_房价预测:", y_predict)
    print("岭回归_均分误差:", error)

    return None


if __name__ == '__main__':
    linear_Regression()
    linear_SGDRegressor()
    linear_Ridge()
