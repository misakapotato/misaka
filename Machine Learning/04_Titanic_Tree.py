import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


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
                                         ignore_index=True) # 去掉年龄为空值的行
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
    # 2.3.分开训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y.astype('int'), random_state=22)
    return x_train, x_test, y_train, y_test

    # 决策树可视化
def show_tree(estimator, feature_name):
    export_graphviz(estimator, out_file="../titanic_tree.dot", feature_names=feature_name)
    return None


def titanic_test():
    # 3.特征工程
    # 3.1.承接数据集
    x_train, x_test, y_train, y_test = load_data()
    # 3.2.特征预处理-标准化
    transfer = DictVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    
    # 4.预估-预估器：决策树
    estimator = DecisionTreeClassifier(criterion="entropy", max_depth=12) # 实例化预估器
    estimator.fit(x_train, y_train)
    show_tree(estimator, transfer.get_feature_names()) # 可视化
    
    # 5.模型评估
    # 5.1.方法1-直接比对
    y_predict = estimator.predict(x_test)
    print("预测值为:", y_predict, "\n真实值为:", y_test, "\n比较结果为:", y_test == y_predict)
    # 5.2.方法2-计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为: ", score)
    return None


if __name__ == '__main__':
    titanic_test()
