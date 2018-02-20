#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: AlexNet
@Product name: PyCharm
@Time: 2018/2/20 10:32
@Author: ldgang
'''
'''
使用tenserflow实现alexNet
有几个新技术：
1，使用了Relu作为激活函数
2，使用dropout削弱过拟合
3，使用最大池化，减小平均池化的模糊问题。同时池化的尺寸大于步进的尺寸，增加了数据的多样性
4，使用CUDA加速运算
5，提出了LRN局部归一标准化
6，数据增强
'''
from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32  #每个batch的样本数
num_batchs = 100  #100个batch

def print_activations(t): #该函数接受一个tensor作为输入，用于展示每一个卷积层或池化层输出tensor的尺寸
    print(t.op.name,' ',t.get_shape().as_list())
def weight_var(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_var(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
'''
inference:构建AlexNet。
卷积-》LRN->池化->卷积-》LRN->池化->卷积-》卷积-》卷积-》池化
'''
def inference(images):
    param = []
    with tf.name_scope('conv1') as scope:  #with...可以将scope内生成的variable自动命名为conv1/xxx，便于区分
        kernal = tf.Variable(tf.truncated_normal([11,11,3,64],dtype = tf.float32,stddev=0.1),name = 'weights') #11*11*3的滤波器，64个
        conv = tf.nn.conv2d(images,kernal, [1,4,4,1],padding='SAME') #卷积，使用SAME卷积，步进是4*4
        biases = tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),trainable=True,name='biases') #卷积层的偏置初始化为0
        bias = tf.nn.bias_add(conv,biases) #卷积的输出加上偏置作为relu函数的输入
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1) #打印激活函数的输出
        param += [kernal,biases] #将参数添加到param中
    lrn1 = tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn1')  #lrn层
    pool1 = tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID', name = 'pool1') #对lrn层的输出进行最大池化，池化尺寸3x3，步长2x2

    with tf.name_scope('conv2') as scope:
        kernal = tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=0.1),name='weigthts')
        conv = tf.nn.conv2d(pool1,kernal,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[192],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(bias,name=scope)
        param += [kernal,biases]
        print_activations(conv2)
    lrn2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn2')
    pool2 = tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool2')
    print_activations(pool2)

    with tf.name_scope('conv3') as scope:
        kernal = tf.Variable(tf.truncated_normal([3,3,192,384],dtype=tf.float32,stddev=0.1,name='weights'))
        conv = tf.nn.conv2d(pool2,kernal,strides=[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[384]),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.relu(bias,name='conv3')
        param += [kernal,biases]
        print_activations(conv3)

    with tf.name_scope('conv4') as scope:
        kernal = tf.Variable(tf.truncated_normal([3,3,384,256],dtype=tf.float32,stddev=0.1,name='weights'))
        conv = tf.nn.conv2d(conv3,kernal,strides=[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv4 = tf.nn.relu(bias,name='conv4')
        param += [kernal,biases]
        print_activations(conv4)

    with tf.name_scope('conv5') as scope:
        kernal = tf.Variable(tf.truncated_normal([3,3,256,256],dtype=tf.float32,stddev=0.1,name='weights'))
        conv = tf.nn.conv2d(conv4,kernal,strides=[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv5 = tf.nn.relu(bias,name='conv5')
        param += [kernal,biases]
        print_activations(conv5)
    pool5 = tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
    print_activations(pool5)
    return pool5,param

def fc_layer(inData,outNum):
    shape = tf.shape(inData)
    inNum = shape[0]*shape[1]*shape[2]
    W_fc = weight_var([inNum, outNum])
    b_fc = bias_var([outNum])
    h_data_flat = tf.reshape(inData, [-1, inNum])
    h_fc = tf.nn.relu(tf.matmul(h_data_flat, W_fc) + b_fc)
    return h_fc

'''
time_tensorflow_run
评估alexNet每轮计算时间的函数。
sess：tensorflow的session
target：需要评估的运算算子
info：测试的名称
'''
def time_tensorflow_run(sess, target, info):
    num_steps_burn_in = 10 #刚开始迭代时存在现存加载，cache命中等问题，跳过，只考量10轮迭代之后的计算时间
    total_duration = 0.0 #计算总时间
    total_duration_squared = 0.0 #平方和计算方差
    for i in range(num_batchs+num_steps_burn_in): #进行的迭代次数
        start_time = time.time() #时间
        sess.run(target) #每次迭代使用session.run(target)执行
        duration = time.time()-start_time
        if i >= num_steps_burn_in: #超过num_steps_burn_in之后，每十轮迭代显示所需要的时间
            if not i%10:
                print("%s: step %d, duration=%.3f"%(datetime.now(),i-num_steps_burn_in,duration))
            total_duration += duration #将耗时加上
            total_duration_squared += duration*duration #将耗时的平方加上
    mn = total_duration/num_batchs #共统计了num_batch次迭代，此处用于求耗时的平均值
    vr = total_duration_squared/num_batchs - mn*mn #方差
    sd = math.sqrt(vr) #标准差
    print("%s: %s across %d steps, %.3f +/- %.3f sec / batch" %(datetime.now(),info,num_batchs,mn,sd))

def run_benchmark():
    with tf.Graph().as_default(): #定义默认的Graph方便后面使用
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,  #利用random_normal函数构造正态分布，标准差为0.1的tensor
                                               image_size, #image_size为图片的尺寸
                                               image_size,3],  #这里的3表示颜色通道数
                                              dtype = tf.float32,
                                              stddev=0.1))
        pool5,param = inference(images)  #构建网络
        init = tf.global_variables_initializer()
        sess = tf.Session() #创建Session
        sess.run(init)
        time_tensorflow_run(sess,pool5,"Forward") #forward评测

        objective = tf.nn.l2_loss(pool5)  #设置优化目标
        grad = tf.gradients(objective,param) #求取相对于loss的所有模型参数的梯度
        time_tensorflow_run(sess,grad, "Forward-backward") #求取时间


if __name__ == "__main__":
    run_benchmark()





