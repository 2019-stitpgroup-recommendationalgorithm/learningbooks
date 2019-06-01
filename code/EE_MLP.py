import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data=pd.read_csv(
                    'u.data',             #设置文件路径
                    sep="\t",                                   #设置读取文件时以空格分隔每列
                    names=["uid","pid","rate","time"]           #设置每列的标题
                 )
K=10                                                             #特征数
beta=0.1                                                       #正则化项系数
alpha=1e-3                                                     #梯度下降步长
steps=100                                                      #梯度下降总次数
flag=0.001                                                      #设置收敛速率小于退出
batch=256                                                 #设置切片大小
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

#-----------------------------------------------------------------------------------


user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')

mlp_P = tf.Variable(tf.random_normal([usernum, K]), dtype=tf.float32)
mlp_Q = tf.Variable(tf.random_normal([itemnum, K]), dtype=tf.float32)
# print(user_id.shape , item_id.shape)
mlp_user_latent_factor = tf.nn.embedding_lookup(mlp_P, user_id)

mlp_item_latent_factor = tf.nn.embedding_lookup(mlp_Q, item_id)

# Gyx = tf.concat([mlp_item_latent_factor, mlp_user_latent_factor], axis=1)
# print(tf.concat([mlp_item_latent_factor, mlp_user_latent_factor], axis=1))
test1 = tf.concat([mlp_item_latent_factor, mlp_user_latent_factor], axis=1)
layer_1 = tf.layers.dense(inputs=tf.concat([mlp_item_latent_factor, mlp_user_latent_factor], axis=1),
                          units= K * 2, kernel_initializer=tf.random_normal_initializer,
                          activation=tf.nn.relu,
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=beta))

layer_2 = tf.layers.dense(inputs=layer_1, units=K * 2, activation=tf.nn.relu,
                          kernel_initializer=tf.random_normal_initializer,
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=beta))

MLP = tf.layers.dense(inputs=layer_2, units=K, activation=tf.nn.relu,
                      kernel_initializer=tf.random_normal_initializer,
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=beta))

pred_y = tf.nn.sigmoid(tf.reduce_sum(MLP, axis=1))

# pred_y = tf.layers.dense(inputs=MLP, units=1, activation=tf.sigmoid)
# pred_y = tf.reduce_sum(pred_y,axis=1)
#-----------------------------------------------------------------------------------
cost=rate - average - b_u - b_i + pred_y

normalpath=tf.square(cost)                                        #得到非正则化项部分
regpath=beta * (tf.nn.l2_loss(Xu - Yi) + tf.nn.l2_loss(b_u) + tf.nn.l2_loss(b_i))            #得到正则化项部分
loss=tf.reduce_sum(normalpath) +regpath                                      #得到损失函数
trainer=tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)          #优化器
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
losses=[]                                                       #每次更新的损失函数列表
rmselist=[]
maelist=[]
user_random = np.random.random_integers(usernum - 1,size =trainnum)
item_random = np.random.random_integers(itemnum - 1,size = trainnum)
# print(user_random , len(item_random))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(steps):
        for i in range(int(trainnum/batch)):
            train_loss_list = []  # 储存各切片的损失函数
            # batch_user = user_random[i * batch:(i + 1) * batch]
            # print(len(batch_user),len(batch_item))
            _,lossbuffer,tmp , tmp2 , tmp3=sess.run([trainer,loss,test1,user_id,item_id],feed_dict={
                                                        user_id : user_random[i * batch:(i + 1) * batch],
                                                        item_id : item_random[i * batch:(i + 1) * batch],
                                                        uid:train.uid.values[i*batch:(i+1)*batch]-1,
                                                        pid:train.pid.values[i*batch:(i+1)*batch]-1,
                                                        rate:train.rate.values[i*batch:(i+1)*batch]
                                                    })
            # print(lossbuffer)
            # print(tmp.shape , tmp3.shape)
            # print(tmp.shape)
            train_loss_list.append(lossbuffer)
        rmse=[]
        mae=[]
        for i in  range(int(testnum/batch)):
           lossbuffer=sess.run(cost,feed_dict={
                                                user_id: user_random[i * batch:(i + 1) * batch],
                                                item_id: item_random[i * batch:(i + 1) * batch],
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
