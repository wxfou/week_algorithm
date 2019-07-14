#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score

class GMM:
    def __init__(self,Data,K,weights = None,means = None,covars = None):
        """
        Data: 训练数据
        K: 高斯分布的个数
        weigths: 每个高斯分布的初始概率（权重）
        means: 高斯分布的均值向量
        covars: 高斯分布的协方差矩阵集合
        """
        self.Data = Data
        self.K = K
        if weights is not None:
            self.weights = weights
        else:
            self.weights  = np.random.rand(self.K)
            self.weights /= np.sum(self.weights) # 归一化
        col = np.shape(self.Data)[1]
        if means is not None:
            self.means = means
        else:
            self.means = []
            for i in range(self.K):
                mean = np.random.rand(col)
                self.means.append(mean)
        if covars is not None:
            self.covars = covars
        else:
            self.covars  = []
            for i in range(self.K):
                cov = np.random.rand(col,col)
                self.covars.append(cov)

    def Gaussian(self,x,mean,cov):
        """
        自定义的高斯分布概率密度函数
        x: 输入数据
        mean: 均值数组
        cov: 协方差矩阵
        返回:x的概率
        """
        dim = np.shape(cov)[0]
        # cov的行列式为零时的措施
        covdet = np.linalg.det(cov + np.eye(dim) * 0.001)
        covinv = np.linalg.inv(cov + np.eye(dim) * 0.001)
        xdiff = (x - mean).reshape((1,dim))
        # 概率密度
        prob = 1.0/(np.power(np.power(2*np.pi,dim)*np.abs(covdet),0.5))*\
               np.exp(-0.5*xdiff.dot(covinv).dot(xdiff.T))[0][0]
        return prob

    def GMM_EM(self):
        loglikelyhood = 0
        oldloglikelyhood = 1
        len,dim = np.shape(self.Data)
        # gamma表示第n个样本属于第k个混合高斯的概率
        gammas = [np.zeros(self.K) for i in range(len)]
        while np.abs(loglikelyhood-oldloglikelyhood) > 0.00000001:
            oldloglikelyhood = loglikelyhood
            # E-step
            for n in range(len):
                # respons是GMM的EM算法中的权重w，即后验概率
                respons = [self.weights[k] * self.Gaussian(self.Data[n], self.means[k], self.covars[k]) for k in range(self.K)]
                respons = np.array(respons)
                sum_respons = np.sum(respons)
                gammas[n] = respons/sum_respons
            # M-step
            for k in range(self.K):
                #nk表示N个样本中有多少属于第k个高斯
                nk = np.sum([gammas[n][k] for n in range(len)])
                # 更新每个高斯分布的概率
                self.weights[k] = 1.0 * nk / len
                # 更新高斯分布的均值
                self.means[k] = (1.0/nk) * np.sum([gammas[n][k] * self.Data[n] for n in range(len)], axis=0)
                xdiffs = self.Data - self.means[k]
                # 更新高斯分布的协方差矩阵
                self.covars[k] = (1.0/nk)*np.sum([gammas[n][k]*xdiffs[n].reshape((dim,1)).dot(xdiffs[n].reshape((1,dim))) for n in range(len)],axis=0)
            loglikelyhood = []
            for n in range(len):
                tmp = [np.sum(self.weights[k]*self.Gaussian(self.Data[n],self.means[k],self.covars[k])) for k in range(self.K)]
                tmp = np.log(np.array(tmp))
                loglikelyhood.append(list(tmp))
            loglikelyhood = np.sum(loglikelyhood)
        for i in range(len):
            gammas[i] = gammas[i]/np.sum(gammas[i])
        self.posibility = gammas
        self.prediction = [np.argmax(gammas[i]) for i in range(len)]

def run_main():
    # 导入Iris数据集
    iris = load_iris()
    label = np.array(iris.target)
    data = np.array(iris.data)
    print("Iris数据集的标签：\n",label)
    # 对数据进行预处理
    data = Normalizer().fit_transform(data)
    # GMM模型
    K = 3
    gmm = GMM(data,K)
    gmm.GMM_EM()
    y_pre = gmm.prediction
    print("预测结果：\n",y_pre)
    print("正确率：\n",accuracy_score(label,y_pre))
if __name__ == '__main__':
    run_main()
