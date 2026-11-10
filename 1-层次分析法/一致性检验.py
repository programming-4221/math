import numpy

#定义矩阵A
A = numpy.array([[1, 5, 3, 2], [1/5, 1, 1/2, 1/3], [1/3, 2, 1, 1/2], [1/2, 3, 2, 1]])
#A = np.array([[1, 2], [1/2, 1]])

#获取A的行与列
n, _ = A.shape

#求出最大特征值与对应的特征向量
eig_values , eig_vectors = numpy.linalg.eig(A)#eig_values是特征值，eig_vectors是特征向量
max_eig = max(eig_values)                     #求最大特征值
max_index = numpy.argmax(eig_values)          #求最大特征值的索引

CI = (max_eig - n) / (n - 1)
RI = [0, 0, 0.52, 0.89, 1.12, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]
#n最多支持n = 15
#n = 2时一定为一致矩阵，所以CI = 0，为了分母为0，第二个数改为接近0的数

if n != 2:
    CR = CI / RI[n - 1]
else:
    CR = 0.0

print(A)
print("一致性指标CI=", CI)
print("一致性比例CR=", CR)

if CR < 0.1:
    print("CR<0.1,right")
else :
    print("CR>=0.1,wrong")



"""

算数平均法

"""

def arithmetic ():

    #每列和ASum
    ASum = numpy.sum(A, axis=0)

    #归一化
    stand_A = A / ASum

    #各列相加到同一行
    ASumr = numpy.sum(stand_A, axis=1)

    #计算权重向量
    wights = ASumr / n

    print(wights)

"""

几何平均法

"""

def geometry ():

    #将A的每一行元素相乘得到一列向量
    prod_A = numpy.prod(A, axis=1)

    #新向量每个分量开n次方
    prod_n_A = numpy.power(prod_A, 1/n)

    #归一化
    re_prod_A = prod_n_A / numpy.sum(prod_n_A)

    print(re_prod_A)

"""

特征值平均法

"""

def eigenvalue ():

    #找出对应的特征向量
    max_vector = eig_vectors[:, max_index]

    #对特征向量做归一化处理,得到权重
    wights_2 = max_vector / numpy.sum(max_vector)

    print(wights_2)

arithmetic()
geometry()
eigenvalue()
