import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data=pd.read_csv(
                    '../TrainDate/ml-100k/u.data',             #设置文件路径
                    sep="\t",                                   #设置读取文件时以空格分隔每列
                    names=["uid","pid","rate","time"]           #设置每列的标题
                 )
K=10                                                             #特征数
beta=0.1                                                       #正则化项系数
alpha=1e-3                                                     #梯度下降步长
steps=1000                                                       #梯度下降总次数
flag=0.001                                                      #设置收敛速率小于退出
batch=1024                                                       #设置切片大小
test_size=0.2                                                   #测试集比例

usernum=data.uid.unique().shape[0]                              #得到用户的数目
itemnum=data.pid.unique().shape[0]                              #得到物品的数目

train,test=train_test_split(data,test_size=test_size)                #得到训练集和测试集,8/2分
testnum=test.shape[0]                                             #测试集总数
trainnum=train.shape[0]                                           #得到训练集总数

average=np.mean(data.rate.values)                              #得到评分的平均值
uid=tf.placeholder(dtype=tf.int32,shape=[None],name="uid")      #用户矩阵切片
pid=tf.placeholder(dtype=tf.int32,shape=[None],name="pid")      #物品矩阵切片
rate=tf.placeholder(dtype=tf.float32,shape=[None],name="rate")  #真实评分矩阵切片

bu=tf.Variable(tf.random_normal([usernum], stddev=0.01))        #定义用户偏差
bi=tf.Variable(tf.random_normal([itemnum], stddev=0.01))        #定义物品偏差
Y=tf.Variable(tf.random_normal([itemnum,K],stddev=0.01))        #定义物品向量
X=tf.Variable(tf.random_normal([usernum,K],stddev=0.01))        #定义用户向量

b_u=tf.nn.embedding_lookup(bu,uid)                      #用户偏差矩阵
b_i=tf.nn.embedding_lookup(bi,pid)                      #物品偏差矩阵
Xu=tf.nn.embedding_lookup(X,uid)                   #得到用户特征行
Yi=tf.nn.embedding_lookup(Y,pid)                   #得到物品特征行

cost=rate - average - b_u - b_i + tf.reduce_sum(tf.square(Xu - Yi),axis=1)
normalpath=tf.square(cost)                                        #得到非正则化项部分
regpath=beta * (tf.nn.l2_loss(Xu - Yi) + tf.nn.l2_loss(b_u) + tf.nn.l2_loss(b_i))            #得到正则化项部分
loss=tf.reduce_sum(normalpath) +regpath                                      #得到损失函数
trainer=tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)          #优化器
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
losses=[]                                                       #每次更新的损失函数列表
rmselist=[]
maelist=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(steps):
        train_loss_list=[]                                            #储存各切片的损失函数
        for i in range(int(trainnum/batch)):
            _,lossbuffer=sess.run([trainer,loss],feed_dict={
                                                        uid:train.uid.values[i*batch:(i+1)*batch]-1,
                                                        pid:train.pid.values[i*batch:(i+1)*batch]-1,
                                                        rate:train.rate.values[i*batch:(i+1)*batch]
                                                    })
            train_loss_list.append(lossbuffer)
        rmse=[]
        mae=[]
        for i in  range(int(testnum/batch)):
           lossbuffer=sess.run(cost,feed_dict={
                                                uid:test.uid.values[i*batch:(i+1)*batch]-1,
                                                pid:test.pid.values[i*batch:(i+1)*batch]-1,
                                                rate:test.rate.values[i*batch:(i+1)*batch]
                                            })
           rmse.append(np.square(lossbuffer))
           mae.append(np.abs(lossbuffer))
        rmse=np.sqrt(np.sum(rmse)/testnum)
        mae=np.sum(mae)/testnum
        if step%20==0:
            losses.append(np.sum(train_loss_list))
            rmselist.append(rmse)
            maelist.append(mae)
        print("rmse:",rmse,"   mae:",mae,losses[-1])
fig1 = plt.figure('LOSS')
x = range(len(losses))
plt.plot(range(len(rmselist)), rmselist, marker='o', label='MASE')
plt.plot(range(len(maelist)), maelist, marker='v', label='MAE')
plt.title('The MovieLens Dataset Learning Curve')
plt.xlabel('Number of Epochs')
plt.ylabel('MASE&MAE')
plt.legend()
plt.grid()
plt.show()
sess.close()