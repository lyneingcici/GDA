import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from scipy.linalg import block_diag


class GDA():
    def __init__(self, X, y, gamma):
        self.X = X
        self.y = y
        self.N = self.X.shape[0]  #训练样本数
        self.labels = np.unique(y)  #标签
        self.nClass = len(self.labels)  #标签数（类数）
        self.nSample = []
        self.nAtt = self.X.shape[1]  ## 特征数
        self.m = np.mean(self.X, axis=0)  ## 样本均值
        self.Mean = self.get_Mean()
        self.B = self.get_B()
        self.W = self.get_W()
        self.K = rbf_kernel(X=self.X, gamma=gamma)
        #B,W,K的矩阵维数
        print("B::", self.B.shape)
        print("W::", self.W.shape)
        print("K::", self.K.shape)
        #np.dot函数WK,BK进行矩阵乘法，得到的结果求逆，然后求矩阵的特征值和特征向量
        self.L, self.U = np.linalg.eig(np.dot(np.linalg.inv(np.dot(self.W, self.K)), np.dot(self.B, self.K)))

    def get_Mean(self):
        #nclass行natt列0矩阵
        Mean = np.zeros((self.nClass, self.nAtt))
        #遍历标签，获取样本的类均值
        for i, lab in enumerate(self.labels):
            idx_list = np.where(self.y == lab)
            Mean[i] = np.mean(self.X[idx_list], axis=0)
            self.nSample.append(len(idx_list))
        return Mean

    def get_B(self):
        Diag = []
        for i, lab in enumerate(self.labels):
            #idx_list存储y == lab的值
            idx_list = np.where(self.y == lab)
            #idx_list大小
            nSub = np.size(idx_list)
            #创建一个nSub*nSub的数组，其值均为1 / nSub
            subMat = np.ones((nSub, nSub)) * (1 / nSub)
            if i == 0:
                Diag = block_diag(subMat)#将subMat转化为对角阵
            else:
                Diag = block_diag(Diag, subMat)#将Diag, subMat转换为对角阵
        Diag = (1 / self.N) * Diag
        B = Diag - (1 / self.N) * np.ones((self.N, self.N))
        return B

    def get_W(self):
        A_M = []
        B_M = []
        for i, lab in enumerate(self.labels):
            idx_list = np.where(self.y == lab)
            nSub = np.size(idx_list)
            aMat = np.eye(nSub) * (1 / nSub)
            bMat = np.ones((nSub, nSub)) * (1 / self.N)
            if i == 0:
                A_M = block_diag(aMat)
                B_M = block_diag(bMat)
            else:
                A_M = block_diag(A_M, aMat)
                B_M = block_diag(B_M, bMat)
        W = A_M - B_M
        return W

    def DR(self, n_component):
        trans_X = []
        #对从小到大排列的L数组进行翻转
        ord_ids = np.flipud(np.argsort(self.L))
        #创建一个列表
        for i in range(n_component):
            tar_ids = ord_ids[i]
            u = self.U[:, tar_ids]
            if trans_X == []:
                trans_X = self.K @ self.U[:, tar_ids] / np.sqrt(u @ self.K @ u)
            else:
                trans_X = np.vstack((trans_X, self.K @ self.U[:, tar_ids] / np.sqrt(u @ self.K @ u)))
        return trans_X.T


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    gamma = 10
    gda = GDA(X=X, y=y, gamma=gamma)
    print(gda.DR(n_component=2))
