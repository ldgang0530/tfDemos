#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: VGGNet-16
@Product name: PyCharm
@Time: 2018/2/20 17:32
@Author: ldgang
'''

from datetime import datetime
import math
import time
import tensorflow as tf


def conv_op(input_op, name, kh, kw, n_out, dh,dw,p):  #创建卷积层，并把本层的参数存入参数列表
    n_in = input_op.get_shape()[-1].value  #通道数
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w", shape=[kh,kw,n_in,n_out],dtype=tf.float32,  #定义核函数，即滤波器
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op,kernel,[1,dh,dw,1],padding="SAME") #卷积
        bias_init_val = tf.constant(0.0,shape=[n_out],dtype=tf.float32) #初始化偏置
        bias=tf.Variable(bias_init_val,name='b') #偏置矩阵
        z = tf.nn.bias_add(conv,bias) #激活函数的输入
        activation = tf.nn.relu(z,name=scope) #Relu激活函数
        p += [kernel,bias]
        return activation
def fc_op(input_op, name, n_out, p):  #定义全连接层
    n_in = input_op.get_shape()[-1].value  #通道数
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',shape=[n_in,n_out],dtype=tf.float32,  #定义核函数
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        bias = tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='b') #定义偏置
        activation = tf.nn.relu_layer(input_op,kernel,bias,name=scope) #Relu层
        p += [kernel,bias]
        return activation
def mpool_op(input_op, name, kh, kw, dh, dw): #定义最大池化层
    return tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding='SAME')

def inference_op(input_op,keep_op):
    p = []
    #第一段卷积网络
    conv1_1 = conv_op(input_op,name='conv1_1',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
    conv1_2 = conv_op(conv1_1,name='conv1_2',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
    pool1_1 = mpool_op(conv1_2,name='pool1_1',kh=2,kw=2,dh=2,dw=2)
    #第二段卷积网络
    conv2_1 = conv_op(pool1_1, name='conv2_1', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1, name='conv2_2', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2_1 = mpool_op(conv2_2, name='pool2_1', kh=2, kw=2, dh=2, dw=2)
    #第三段卷积网络
    conv3_1 = conv_op(pool2_1, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dh=1,dw=1,p=p)
    pool3_1 = mpool_op(conv3_3, name='pool3_1', kh=2, kw=2, dh=2, dw=2)
    #第四段卷积网络
    conv4_1 = conv_op(pool3_1, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4_1 = mpool_op(conv4_3, name='pool4_1', kh=2, kw=2, dh=2, dw=2)
    #第五段卷积网络
    conv5_1 = conv_op(pool4_1, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5_1 = mpool_op(conv5_3, name='pool5_1', kh=2, kw=2, dh=2, dw=2)

    #对卷积网络的输出扁平化处理
    poolShape = pool5_1.get_shape()
    flat_shape = poolShape[0].value*poolShape[1].value*poolShape[2].value
    resh = tf.reshape(pool5_1,[-1,flat_shape],name='resh')
    #两个全连接层,4096个隐藏点
    fc6 = fc_op(resh,name='fc6',n_out=4096,p=p)
    fc6_drop = tf.nn.dropout(fc6,keep_op,name='fc6_drop')
    fc7 = fc_op(fc6_drop,name='fc7',n_out=4096,p=p)
    fc7_drop = tf.nn.dropout(fc7,keep_op,name='fc7_drop')
    #全连接层，1000个输出节点，使用softmax进行处理得到分类输出概率
    fc8 = fc_op(fc7_drop,name='fc8',n_out=1000,p=p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax,1)
    return predictions,softmax,fc8,p

def time_tensorflow_run(sess, target, feed, info):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_sqared = 0.0
    for i in range(num_batches+num_steps_burn_in):
        start_time = time.time()
        sess.run(target, feed_dict=feed)  #运行计算图
        duration = time.time()-start_time
        if i >= num_steps_burn_in:
            if not i%10:
                print("%s: step %d,duration=%.3f" %(datetime.now(),i-num_steps_burn_in,duration))
            total_duration += duration
            total_duration_sqared += duration*duration
    mn = total_duration/num_batches
    vr = total_duration_sqared/num_batches - mn*mn
    sd = math.sqrt(vr)
    print("%s: %s across %d steps, %.3f +/- %.3f sec / batch" %(datetime.now(), info, num_batches, mn, sd))

def run_benchmark():
    with tf.Graph().as_default():
        image_size=224
        images = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],dtype=tf.float32,stddev=0.1))
        keep_prob = tf.placeholder(tf.float32)
        predictions,softmax,fc8,p= inference_op(images,keep_prob)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        time_tensorflow_run(sess, predictions, {keep_prob:1.0}, "Forward")
        objective = tf.nn.l2_loss(fc8)
        grd = tf.gradients(objective,p)
        time_tensorflow_run(sess, grad, {keep_prob:0.5}, "Forward-backward")

if __name__ == "__main__":
    batch_size = 32
    num_batches = 100
    run_benchmark()