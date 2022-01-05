
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
# from sklearn_learning.model_load_store.Util_model import *

def load_data():
    column_name = ['Sample code number', 'Clump Thickness',
                   'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
                   'Single Epithelial Cell Size',
                   'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']

    # 1.获取数据
    original_data = pd.read_csv("./cancer/breast-cancer-wisconsin.data", names=column_name)
        # 第二种方式-直接下载
        # path = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
        # original_data = pd.read_csv(path, names=column_name)
    # 2.数据处理
    # 2.1.缺失值处理
    # 2.1.1.先替换 ? 为 nan
    data = original_data.replace(to_replace="?", value=np.nan)
    # 2.1.2.检测是否还有缺失值
    data.dropna(inplace=True)
    print("检测是否还有缺失值(全为false表示没有缺失值)\n", data.isnull().any())

    # 2.2. 提取认为有用的特征
    x = data.iloc[:, 1:-1]  # 从第一列到倒数第二列的“column字段”都要，且每行都要
    y = data["Class"]
    
    # 2.3.分开训练集和测试集  
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    
    return x_train, x_test, y_train, y_test


def logic_Regression():

    # 3.特征工程
    # 3.1.承接数据集
    x_train, x_test, y_train, y_test = load_data()
    # 3.2.特征预处理-标准化 
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.预估-预估器：逻辑回归
    estimator = LogisticRegression()  # 默认参数
    estimator.fit(x_train, y_train)
    print("逻辑回归_系数为: ", estimator.coef_)
    print("逻辑回归_截距为:", estimator.intercept_)

    # store_model(estimator, "logic_regression_model01.pkl")  # 保存模型
    # estimator = load_model("logic_regression_model01.pkl")  # 加载模型

    # 5.模型评估
    # 5.1.传统方式
    y_predict = estimator.predict(x_test)
    print("逻辑回归_预测结果", y_predict)
    print("逻辑回归_预测结果对比:", y_test == y_predict)
    score = estimator.score(x_test, y_test)
    print("准确率为:", score)
    # 2是良性的 4是恶性的

    # 5.2.F1-score评估
    Score = classification_report(y_test, y_predict, labels=[2, 4],
                                  target_names=["良性", "恶性"])
    print("查看精确率,召回率,F1-score\n", Score)
    
    # 5.3.ROC & AUC评估
    y_true = np.where(y_test > 3, 1, 0)  # 把目标值转换为0,1表示
    return_value = roc_auc_score(y_true, y_predict)
    print("ROC曲线和AUC返回值为(三角形面积)", return_value)
    fpr, tpr, thresholds = roc_curve(y_true, y_predict)
    plt.plot(fpr, tpr)
    plt.show()
    
    return None


if __name__ == '__main__':
    logic_Regression()
