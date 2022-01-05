
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score  

# 降维
def decomposition():
    # 1.引入数据
    order_product = pd.read_csv("./instacart/order_products__prior.csv")
    aisles = pd.read_csv("./instacart/aisles.csv")
    orders = pd.read_csv("./instacart/orders.csv")
    products = pd.read_csv("./instacart/products.csv")

    # 2.合并表
    table_1 = pd.merge(aisles, products, on=["aisle_id", "aisle_id"])  # 让aisles 和product_id一起
    table_2 = pd.merge(table_1, order_product, on=["product_id", "product_id"])
    table_3 = pd.merge(table_2, orders, on=["order_id", "order_id"])

    # 3.数据交叉表
    result_table = pd.crosstab(table_3["user_id"], table_3["aisle"])
    # print(result_table)  # 处理后的最终数据
    
    # 4.PCA降维
    result_table = result_table[:10000]  # 只取10000组数据, 防止后面预测的时候内存爆炸
    transfer = PCA(n_components=0.95)
    data_new = transfer.fit_transform(result_table)
    print("数据样本", data_new.shape)
    print("PCA降维结果:\n", data_new)
    return data_new

# 聚类
def KMeans_test():
    # 1.获取数据
    data = decomposition()

    # 2.预估-预估器：K-means算法
    estimator = KMeans(n_clusters=3, init='k-means++')
    estimator.fit(data)
    predict = estimator.predict(data)  # 分成的组, 0 1 2
    print("展示前300个用户的类别", predict[:300])

    # 3.模型评估:
    """
    引入轮廓系数分析和对应公式, 见截图
    """
    score = silhouette_score(data, predict)  # 此处容易爆内存
    print("模型轮廓系数为(1 最好, -1 最差):", score)

    return None


if __name__ == '__main__':
    KMeans_test()

