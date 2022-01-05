from sklearn.datasets import fetch_20newsgroups, load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

"""
朴素贝叶斯算法: 朴素: 假设特征和特征之间是相互独立的
朴素贝叶斯算法经常用于文本分类, 因为文章转换成机器学习算法识别的数据是以单词为特征的
学习算法要点包括但不限于拉普拉斯平滑系数,朴素贝叶斯, 简单概率论
"""


def load_data():
    # 1.获取数据
    all_data = fetch_20newsgroups(subset="all")  # 下载新闻
    # all_data = load_files(container_path="./20news-bydate-train") # 本地获取数据
    
    # 2.数据处理-分开训练集和测试集
    x_train, x_test, y_train, y_test = \
        train_test_split(all_data.data, all_data.target, \
                         test_size=0.2, random_state=1853799) 
            
    # 3.特征工程
    # 3.1.特征抽取-文本特征抽取
    transfer = TfidfVectorizer()
    # 3.2.特征预处理-标准化
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)  # 测试集不用fit, 因为要保持和训练集处理方式一致
    
    return x_train, x_test, y_train, y_test


def naive_bayes_test():  
    # 4.预估
    
    # 4.0.承接数据集
    x_train, x_test, y_train, y_test = load_data()
    
    # 4.1.预估器-朴素贝叶斯算法
    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)
    
    # 5.模型评估
    # 5.1.方法1-直接比对
    y_predict = estimator.predict(x_test)
    print("预测值为:", y_predict, "\n真实值为:", y_test, "\n比较结果为:", y_test == y_predict)
    # 5.2.方法2-计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为: ", score)
    
    return None


if __name__ == '__main__':
    naive_bayes_test()