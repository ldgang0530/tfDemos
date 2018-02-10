
from tensorflow.examples.tutorials.mnist import input_data #对mnist数据加载，tensorflow提供了封装接口，可将数据加载成需要的格式
MNIST_data_folder="MNIST_data/"  #注因远程下载失败，这是下载到本地的MNIST数据集路径
mnist = input_data.read_data_sets(MNIST_data_folder, one_hot=True) #读取数据集
print(mnist.train.images.shape, mnist.train.labels.shape) #打印train，test,validation等数据的维度
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

import tensorflow as tf  #导入tensorflow包
sess = tf.InteractiveSession()  #创建一个默认的session
x = tf.placeholder(tf.float32,[None,784]) #输入数据的地方，第一个是数据类型，第二个是代表tensor的shape
W = tf.Variable(tf.zeros([784,10])) #variable类型，存储模型参数，会持久化保存
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W)+b)  #softmax软分类器回归。matmul是矩阵乘法

y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1])) #交叉熵，代价函数
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) #梯度下降，学习率0.5，最小化交叉熵
tf.global_variables_initializer().run() #全局参数初始化

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100) #每次随机抽取100个样本，并feed给placeholder
    train_step.run({x:batch_xs,y_:batch_ys}) #训练

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1)) #argmax用于寻找最大值的序号，这里序号就对应了真实的数字
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #预测正确率
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))


