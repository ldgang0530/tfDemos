#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: tf_verificationCode
@Product name: PyCharm
@Time: 2018/3/6 16:42
@Author: ldgang
'''
import numpy as np
import os

from func import *
#from CNN import cnn
import random
def get_batch(iter=0, batchSize=128, fileNameList=[], trainFlag=True): #获取一个batch
    batchX = np.zeros([batchSize,IMAGE_HEIGHT*IMAGE_WIDTH])
    batchY = np.zeros([batchSize,CODE_NUM*charSetLen])
    totalImageNum = len(fileNameList)
    iterNum = iter*batchSize
    filePath=''
    if trainFlag:
        filePath = IMAGE_PATH
    else:
        filePath = TEST_PATH
    for i in range(batchSize):
        #batchX
        fileName = fileNameList[(iterNum+i)%totalImageNum]  #若超出了总数目就从头开始取
        image = get_image_data(filePath+fileName)
        image = image.eval()
        batchX[i, :] = image.flatten()/255
        #batchY
        batchY[i, :] = get_image_label(fileName)
    return batchX, batchY

def cnn(X,keep_prob):
    def weight_var(shape, name='weight'): #权重矩阵
        init = tf.truncated_normal(shape, stddev=0.1)
        var = tf.Variable(initial_value = init, name=name)
        return var
    def bias_var(shape, name='bias'): #偏置矩阵
        init = tf.truncated_normal(shape, stddev = 0.1)
        var = tf.Variable(initial_value=init, name=name)
        return var

    def conv2d(x, w,name='conv2d'):  #卷积
        return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', name=name)

    def max_pool(value, name='max_pool'): #最大池化
        return tf.nn.max_pool(value, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME', name=name)

    #输入层
    x_input = tf.reshape(X, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name='x-input')
    #第一层 60*160*1 ,图片转换为灰度图片
    w1 = weight_var([5,5,1,32],name='w1')
    b1 = bias_var([32],name='b1')
    conv1 = tf.nn.relu(conv2d(x_input,w1,name='conv1')+b1)  #Relu作为激活函数
    conv1_pool = max_pool(conv1,name='max_pool1') #池化
    conv1_drop = tf.nn.dropout(conv1_pool,keep_prob=keep_prob) #dropout
    #第二层 #30*80*32
    w2 = weight_var([5,5,32,64],name='w2')
    b2 = bias_var([64],name='b2')
    conv2 = tf.nn.relu(conv2d(conv1_drop, w2, name='conv2') + b2)  # Relu作为激活函数
    conv2_pool = max_pool(conv2, name='max_pool2')  # 池化
    conv2_drop = tf.nn.dropout(conv2_pool, keep_prob=keep_prob)  # dropout
    #第三层 #15*40*64
    w3 = weight_var([5, 5, 64, 64], name='w3')
    b3 = bias_var([64], name='b3')
    conv3 = tf.nn.relu(conv2d(conv2_drop, w3, name='conv3') + b3)  # Relu作为激活函数
    conv3_pool = max_pool(conv3, name='max_pool3')  # 池化
    conv3_drop = tf.nn.dropout(conv3_pool, keep_prob=keep_prob)  # dropout
    #全连接层
    w4 = weight_var([8*20*64, 1024], name='w4')
    b4 = bias_var([1024], name='b4')
    fc1 = tf.reshape(conv3_drop, [-1, 8* 20 * 64])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, w4), b4))
    fc1 = tf.nn.dropout(fc1, keep_prob)
    #输出层
    w_out = weight_var([1024, CODE_NUM * len(globCharSet)], 'w_out')
    b_out = bias_var([CODE_NUM * len(globCharSet)], 'b_out')
    out = tf.add(tf.matmul(fc1, w_out), b_out, 'output')
    return out
def train():
    import time
    start_time = time.time()
    fileNameList = [filePath.split('/')[-1] for filePath in os.listdir(IMAGE_PATH)]  # 从图像路径下读取图像文件
    fileNum = len(fileNameList)
    random.seed(start_time)
    random.shuffle(fileNameList)
    trainNum = int(fileNum*VALIDATE_PRECENT)

    trainImageNameList = fileNameList[0:trainNum]
    validateImageNameList = fileNameList[trainNum:]
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH], name='dataInput')
    Y = tf.placeholder(tf.float32, [None, CODE_NUM*charSetLen], name='labelInput')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')

    output = cnn(X, keep_prob)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, CODE_NUM, charSetLen], name='predict')
    labels = tf.reshape(Y, [-1, CODE_NUM, charSetLen], name='labels')

    p_max_idx = tf.argmax(predict, 2, name='predictMaxIndex')
    l_max_idx = tf.argmax(labels,2,name='labelMaxIndex')
    equalVec = tf.equal(p_max_idx, l_max_idx)
    accuracy = tf.reduce_mean(tf.cast(equalVec, tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps = 0
        targetArrived = False
        for epoch in range(EPOCH_NUM):
            train_data, train_label = get_batch(steps,64,trainImageNameList, trainFlag=True)
            sess.run(optimizer, feed_dict={X: train_data, Y: train_label, keep_prob: 0.75})
            if steps % 100 == 0:
                test_data, test_label = get_batch(steps,50, validateImageNameList, trainFlag=True)
                acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label, keep_prob: 1.0})
                print("steps=%d, accuracy=%f" % (steps, acc))
                if acc > 0.99:
                    targetArrived = True
                    break
            steps += 1
        if targetArrived:
            print("Accuracy target arrived")
        else:
            print("Accuracy target doesn't arrive")
        saver.save(sess, MODEL_PATH + "crack_captcha.model", global_step=steps)
'''
if __name__ == '__main__':
    train()
    print("haha")
'''
