
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

'''
xavier初始化----用于初始化权重
若权重过小，信号在每层间传递时逐渐缩小难以起作用
若权重过大，传递时逐渐放大并导致发散和失效
xaiver初始化就是让权重被初始化得不打不小，正好合适，其根据输入输出节点数目自动调整
xaiver使均值为0，方差为2/(n1+n2)
'''
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant*np.sqrt(6.0/(fan_in+fan_out)) #
    high = -low
    # 创建[-sqrt(6/(n1+n2), sqrt(6/(n1+n2)))]的均匀分布
    return tf.random_uniform((fan_in,fan_out),minval=low, maxval=high,dtype=tf.float32)
'''
n_input:输入变量数
n_hidden:隐含层节点数
transfer_function：隐含层激活函数
optimizer:优化器
scale:高斯噪声系数
'''
class AdditiveGaussianNoiseAutoEncoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(),scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        self.weights = self._initalize_weights()
        self.x = tf.placeholder(tf.float32,[None, n_input])  #输入X
        self.hidden = self.transfer(tf.add(tf.matmul(self.x+scale*tf.random_normal((n_input,)),
                                                     self.weights['W1']),self.weights['b1'])) #self.hidden=Relu(W1*X+B1)，得到的是激活函数的输出
        #self.hidden2 = self.transfer(tf.add(tf.matmul(self.hidden,self.weights['W3']),self.weights['b3']))
        self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['W2']),self.weights['b2']) #重建，W2*self.hidden+B2
        self.cost = 0.5+tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0)) #定义代价函数，平方误差和。substract是做减法
        self.optimizer = optimizer.minimize(self.cost) #默认选用Adam算法优化，最小化代价函数
        init = tf.global_variables_initializer() #初始化权重变量
        self.sess = tf.Session() #创建session
        self.sess.run(init) #初始化模型参数
    def _initalize_weights(self):
        all_weights = dict()
        all_weights['W1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden)) #定义W1变量，
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32)) #定义B1
       # all_weights['W3'] = tf.Variable(tf.zeros([self.n_hidden, self.n_hidden], dtype=tf.float32))  # 定义W3
      #  all_weights['b3'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))  # 定义B3
        all_weights['W2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32)) #定义W2
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype=tf.float32)) #定义B2
        return all_weights #返回所有的权重系数
    def partial_fit(self,X):  #该函数是用一个batch数据进行训练并返回当前的损失
        cost,opt=self.sess.run((self.cost,self.optimizer),feed_dict={self.x: X, self.scale:self.training_scale})
        return cost #返回当前batch的损失
    def calc_total_cost(self,X): #统计损失，该函数是在字编码器训练完毕后，在测试集上对模型性能进行评测室会用到
        return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})
    def transform(self,X): #返回自编码器隐含层的输出结果，提供了一个接口来获取抽象后的特征。
        return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})
    def generate(self,hidden=None): #将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction,feed_dict = {self.hidden:hidden})
    def reconstruct(self,X):  #整体运行一遍复原过程，包括提取高阶特征和通过高阶特征复原数据
        return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.training_scale})
    def getWeights(self): #获取w1
        return self.sess.run(self.weights['W1'])
    def getBiases(self): #获取b1
        return self.sess.run(self.weights['b1'])

def standard_scale(X_train,X_test): # 对测试和训练数据进行标准化处理
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train,X_test
def get_random_block_from_data(data,batch_size): #获取随机block数据的函数。
    start_index = np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True) #加载MNIST数据集
    X_train,X_test = standard_scale(mnist.train.images,mnist.test.images) #标准化处理
    n_samples = int(mnist.train.num_examples) #获取总的训练练样本
    training_epochs = 20 #最大训练的轮数
    batch_size = 128 #每次迭代包含的数据的样本大小
    display_step = 1 #这里表示迭代几次，输出结果。设为1表示每一次迭代(遍历完数据集)，都显示结果
    #创建一个AGN自编器的实例，输入节点784，隐含层节点数200，隐含层的激活函数是softplus，优化器采用Adam，学习率0.001，噪声系数0.01
    autoencoder = AdditiveGaussianNoiseAutoEncoder(
                    n_input = 784,
                    n_hidden=200,
                    transfer_function=tf.nn.softplus,
                    optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                    scale = 0.01)
    for epoch in range(training_epochs): #循环epochs轮
        avg_cost = 0.0
        total_batch = int(n_samples/batch_size) #mini-batch，数据集分成了多少个，每个自己大小为batch_size
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train,batch_size) #从数据集随机抽取block个数据
            cost = autoencoder.partial_fit(batch_xs) #计算一个batch的损失
            avg_cost += cost/n_samples*batch_size #统计平均损失
        if epoch%display_step == 0:
            print("Epoch:","%04d"%(epoch+1),'cost=',"{:.9f}".format(avg_cost))
    print('Total cost:' +str(autoencoder.calc_total_cost(X_test))) #在测试集上运用，得到损失的结果





