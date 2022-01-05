# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def decomposition_test():

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
    transfer = PCA(n_components=0.95)
    data_new = transfer.fit_transform(result_table)
    print("PCA降维结果:\n", data_new)
    return None


if __name__ == '__main__':
    decomposition_test()
