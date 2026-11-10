import numpy as np

RI = [0, 0, 0.52, 0.89, 1.12, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]


class Matrix:
    def __init__(self, A):
        self.A = np.array(A, dtype=float)
        self.n = self.A.shape[0]

    def print_result(self, A, CI, CR):
        print("判断矩阵：")
        print(A)
        print("一致性指标CI =", CI)
        print("一致性比例CR =", CR)

        if CR < 0.1:
            print("CR < 0.1, 一致性检验通过")
        else:
            print("CR >= 0.1, 一致性检验不通过")

    def arithmetic(self):
        """算术平均法求权重"""
        A = self.A
        n = self.n

        # 每列和
        ASum = np.sum(A, axis=0)

        # 归一化
        stand_A = A / ASum

        # 各列相加到同一行
        ASumr = np.sum(stand_A, axis=1)

        # 计算权重向量
        weights = ASumr / n

        print("算术平均法权重：", weights)
        return weights

    def geometry(self):
        """几何平均法求权重"""
        A = self.A
        n = self.n

        # 每一行元素相乘
        prod_A = np.prod(A, axis=1)

        # 每个分量开 n 次方
        prod_n_A = np.power(prod_A, 1 / n)

        # 归一化
        weights = prod_n_A / np.sum(prod_n_A)

        print("几何平均法权重：", weights)
        return weights

    def eigenvalue(self):
        """特征值法求权重，并做一致性检验"""
        A = self.A
        n = self.n

        eig_values, eig_vectors = np.linalg.eig(A)

        # 最大特征值对应的索引
        max_index = np.argmax(eig_values.real)
        max_eig = eig_values[max_index].real

        # 最大特征值对应的特征向量
        max_vector = np.abs(eig_vectors[:, max_index].real)

        # 归一化得到权重
        weights = max_vector / np.sum(max_vector)

        print("特征值法权重：", weights)
        print("最大特征值：", max_eig)

        # 一致性指标 CI
        if n > 1:
            CI = (max_eig - n) / (n - 1)
        else:
            CI = 0.0

        # 一致性比例 CR
        if n == 2:
            CR = 0.0
        elif n > 2 and RI[n - 1] != 0:
            CR = CI / RI[n - 1]
        else:
            CR = 0.0

        self.print_result(A, CI, CR)

        return weights, CI, CR, max_eig


if __name__ == "__main__":
    A = np.array([
        [1, 5, 3, 2],
        [1/5, 1, 1/2, 1/3],
        [1/3, 2, 1, 1/2],
        [1/2, 3, 2, 1]
    ])

    m = Matrix(A)

    m.arithmetic()
    m.geometry()
    m.eigenvalue()