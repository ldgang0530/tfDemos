#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: CNN_simple
@Product name: PyCharm
@Time: 2018/2/19 15:41
@Author: ldgang
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)  #读取数据集
sess = tf.InteractiveSession() #创建session

def weight_var(shape):  #权重矩阵
    initial = tf.truncated_normal(shape,stddev=0.1) #利用随机噪声初始化矩阵
    return tf.Variable(initial)
def bias_var(shape):
    initial = tf.constant(0.1,shape = shape)  #设置偏置
    return tf.Variable(initial)
def conv2d(x,W):  #卷积，SAME卷积，在图片的外侧天价了像素，输出时的长宽不发生变化
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x): #池化
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding = 'SAME')
x = tf.placeholder(tf.float32,[None,784]) #定义x为占位符
y_ = tf.placeholder(tf.float32,[None,10]) #定义y_为占位符
x_image = tf.reshape(x,[-1,28,28,1]) #将1*784转换为28*28

#第一层
W_conv1 = weight_var([5,5,1,32]) #第一层卷积的权重矩阵，也就是滤波器。5x5的卷积，1个通道，32个不同的卷积核
b_conv1 = bias_var([32]) #第一层卷积的偏置
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) #隐藏层的Relu函数输出 :28x28*32
h_pool1 = max_pool_2x2(h_conv1) #卷积之后，进行池化:14*14*32
#第二层
W_conv2 = weight_var([5,5,32,64]) #第二层卷积的配置。5x5的卷积，32个通道，64个卷积核。
b_conv2 = bias_var([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) #14*14*64
h_pool2 = max_pool_2x2(h_conv2) #池化操作：7*7*64
#全连接层
W_fc1 = weight_var([7*7*64,1024]) #全连接层的权重矩阵。1024表示设置了全连接层有1024个节点
b_fc1 = bias_var([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64]) #将三维的矩阵，转换为一维
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)  #全连接矩阵使用Relu函数
keep_prob = tf.placeholder(tf.float32) #dropout的保留率
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob) #dropout设置
#全连接层
W_fc2 = weight_var([1024,10]) #输入有1024个节点，输出有10个节点
b_fc2 = bias_var([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2) #采用softmax分类
#代价函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1])) #交叉熵为代价函数
#优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #使用Adam优化器

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1)) #判断是否正确
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #求平均值，此处其实也就是求得了正确率
tf.global_variables_initializer().run() #varible变量初始化
#训练
for i in range(20000): #迭代两万次
    batch = mnist.train.next_batch(50) #每个batch有50个样本
    if i%100==0: #每一百次输出正确率
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d,training accuracy %g" %(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5}) #训练，设置输入输出和keep_prob
#测试
print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))#测试集的正确率
