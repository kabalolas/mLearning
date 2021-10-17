import numpy as np
from  numpy.linalg import cholesky
import matplotlib.pyplot as plt
import tensorflow as tf

############生成测试数据###############
sampleNo = 1000;#数据数量
mu =3
# 二维正态分布
mu = np.array([[1, 5]])
Sigma = np.array([[1, 0.5], [1.5, 3]])
R = cholesky(Sigma)
srcdata= np.dot(np.random.randn(sampleNo, 2), R) + mu
plt.plot(srcdata[:,0],srcdata[:,1],'bo')