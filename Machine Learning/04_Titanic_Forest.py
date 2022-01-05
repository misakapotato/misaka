import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split

"""
# 随机森林就是多个树, 最后通过投票选择多数的那个决策
# 随机有两种方式
# 1: 每一个树训练集不同
# 2: 需要训练的特征进行随机分配 从特定的特征集里面抽取一些特征来分配
"""


def load_data():
    # 1.获取数据
    data = pd.read_csv("./titanic/titanic.csv")
    titanic = data.copy()
    
    # 2.数据处理
    # 2.1.只提取认为有用的特征
    data_used = titanic[["pclass", "age", "sex", "survived"]] 
    real_data = pd.DataFrame(columns=["pclass", "age", "sex", "survived"])
    for row in data_used.values:
        if not np.isnan(row[1]):
            real_data = real_data.append([{'pclass': row[0], 'age': row[1],
                                           'sex': row[2], 'survived': row[3]}],
                                         ignore_index=True)
    # 2.2.转换成字典格式
    x = real_data[["pclass", "age", "sex"]].to_dict(orient="records")
    y = real_data["survived"]

    """
    以上是方法一: 过滤掉空的值的数据组
    以下是方法二: 对空数据设置一个非0值
    x = titanic[["pclass", "age", "sex"]]  
    y = titanic["survived"]  # 目标值
    x["age"].fillna(x["age"].mean(), inplace=True) # 把平均值填补进去
    x = x.to_dict(orient="records")
    """

    x_train, x_test, y_train, y_test = train_test_split(x, y.astype('int'), random_state=22)
    return x_train, x_test, y_train, y_test


def titanic_ramdo_test():
    # 3.特征工程
    # 3.1.承接数据集
    x_train, x_test, y_train, y_test = load_data()
    # 3.2.特征预处理-标准化
    transfer = DictVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    
    # 4.预估-预估器：随机森林  
    estimator = RandomForestClassifier() # 实例化预估器（默认bootstrap为true,即有放回随机抽样）

    param_dict = {"n_estimators": [120, 200, 300, 500, 800, 1200],
                  "max_depth": [5, 8, 15, 25, 30]} # 随机森林参数设置    
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3) # 参数在网格搜索下发挥作用
    
    estimator.fit(x_train, y_train)  # 训练
    
    # 5.模型评估
    # 5.1.方法1-直接比对
    y_predict = estimator.predict(x_test)
    print("预测值为:", y_predict, "\n真实值为:", y_test, "\n比较结果为:", y_test == y_predict)
    # 5.2.方法2-计算准确率
    score = estimator.score(x_train, y_train)
    print("准确率为: ", score)
    # 6.输出最佳参数
    print("最佳参数:\n", estimator.best_params_)
    print("最佳结果:\n", estimator.best_score_)
    print("最佳估计器:\n", estimator.best_estimator_)
    print("交叉验证结果:\n", estimator.cv_results_)

    return None


if __name__ == '__main__':
    titanic_ramdo_test()
    
    
