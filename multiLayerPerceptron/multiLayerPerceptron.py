#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: multiLayerPerceptron
@Product name: PyCharm
@Time: 2018/2/19 10:30
@Author: ldgang
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

inNum = 784  #输入节点数
hiNum = 300  #隐含层的输出节点数
W1 = tf.Variable(tf.truncated_normal([inNum,hiNum],stddev = 0.1)) #隐含层的权重，初始化为ie阶段的正态分布，标准层0.1
b1 = tf.Variable(tf.zeros([hiNum])) #隐含层的偏置，初始化为0
W2 = tf.Variable(tf.zeros([hiNum,10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32,[None,inNum]) #输入数据
keepProb = tf.placeholder(tf.float32) #定义dropout的保留率
hid1 = tf.nn.relu(tf.matmul(x,W1)+b1) #Relu激活函数
hid1Drop = tf.nn.dropout(hid1,keepProb)  #dropout函数
y = tf.nn.softmax(tf.matmul(hid1Drop,W2)+b2) #softmax分类，并输出
y_ = tf.placeholder(tf.float32,[None,10])
crossEntropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))  #定义代价函数，交叉熵
trainStep = tf.train.AdagradOptimizer(0.3).minimize(crossEntropy) #选定优化器，并为优化器指定代价函数

tf.global_variables_initializer().run() #初始化
for i in range(3000):  #共采用300000个样本，每次100个样本
    batch_xs, batch_ys = mnist.train.next_batch(100)  #
    trainStep.run({x:batch_xs,y_:batch_ys,keepProb:0.75}) #dropout的保留率

correctPrediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))  #
accuracy = tf.reduce_mean(tf.cast(correctPrediction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keepProb:1.0})) #测试时，不需要dropout