#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: tfFlow
@Product name: PyCharm
@Time: 2018/2/14 9:23
@Author: ldgang
步骤：
1,定义算法公式，softmax(wx+b)
2,定义代价函数，loss = -sum(y_*log(y))
3,选定优化器，gradientDescentOptimizer,并给优化器制定代价函数
4，训练
5，测试
6，使用

'''

'''
主要介绍了tensorflow的基本使用过程
以代价函数为 J= (w-6)^2 = w^2-12*w+36 为例
'''
import numpy as np
import tensorflow as tf

def costFunc(X,W): #定义代价函数，X为代价函数的系数
    costValue = X[0]*W**2 - X[1]*W + X[2]
    return costValue

def trainFun(costValue,learningRate = 0.01): #默认学习率为0.01
    train = tf.train.GradientDescentOptimizer(learningRate).minimize(costValue)
    return train
if __name__ == "__main__":
    coeff = np.array([[1.], [12.], [36.]])
    W = tf.Variable(0,dtype = tf.float32) #Variable变量
    X = tf.placeholder(tf.float32,[3,1]) #placeHolder变量
    costValue = costFunc(X,W)  #代价函数
    train = trainFun(costValue) #训练算法，即定义了 学习的算法，此处用梯度下降，调用一次意味着迭代一次

    init = tf.global_variables_initializer() #变量初始化
    session = tf.Session() #计算图
    session.run(init) #初始化

    iterNum = 1000 #设置超参，梯度下降的迭代次数
    for i in range(iterNum):
        session.run(train,feed_dict={X:coeff}) #训练
    print(session.run(W))