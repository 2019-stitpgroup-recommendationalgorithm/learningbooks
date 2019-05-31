import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data=pd.read_csv(
                    '../TrainDate/ml-100k/u.data',             #�����ļ�·��
                    sep="\t",                                   #���ö�ȡ�ļ�ʱ�Կո�ָ�ÿ��
                    names=["uid","pid","rate","time"]           #����ÿ�еı���
                 )
K=10                                                             #������
beta=0.1                                                       #������ϵ��
alpha=1e-3                                                     #�ݶ��½�����
steps=100                                                      #�ݶ��½��ܴ���
flag=0.001                                                      #������������С���˳�
batch=1024                                                       #������Ƭ��С
test_size=0.2                                                   #���Լ�����

usernum=data.uid.unique().shape[0]                              #�õ��û�����Ŀ
itemnum=data.pid.unique().shape[0]                              #�õ���Ʒ����Ŀ

train,test=train_test_split(data,test_size=test_size)                #�õ�ѵ�����Ͳ��Լ�,8/2��
testnum=test.shape[0]                                             #���Լ�����
trainnum=train.shape[0]                                           #�õ�ѵ��������

average=np.mean(data.rate.values)                              #�õ����ֵ�ƽ��ֵ
uid=tf.placeholder(dtype=tf.int32,shape=[None],name="uid")      #�û�������Ƭ
pid=tf.placeholder(dtype=tf.int32,shape=[None],name="pid")      #��Ʒ������Ƭ
rate=tf.placeholder(dtype=tf.float32,shape=[None],name="rate")  #��ʵ���־�����Ƭ

bu=tf.Variable(tf.random_normal([usernum], stddev=0.01))        #�����û�ƫ��
bi=tf.Variable(tf.random_normal([itemnum], stddev=0.01))        #������Ʒƫ��
Y=tf.Variable(tf.random_normal([itemnum,K],stddev=0.01))        #������Ʒ����
X=tf.Variable(tf.random_normal([usernum,K],stddev=0.01))        #�����û�����

b_u=tf.nn.embedding_lookup(bu,uid)                      #�û�ƫ�����
b_i=tf.nn.embedding_lookup(bi,pid)                      #��Ʒƫ�����
Xu=tf.nn.embedding_lookup(X,uid)                   #�õ��û�������
Yi=tf.nn.embedding_lookup(Y,pid)                   #�õ���Ʒ������

user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')

mlp_P = tf.Variable(tf.random_normal([usernum, K]), dtype=tf.float32)
mlp_Q = tf.Variable(tf.random_normal([itemnum, K]), dtype=tf.float32)

mlp_user_latent_factor = tf.nn.embedding_lookup(mlp_P, user_id)
mlp_item_latent_factor = tf.nn.embedding_lookup(mlp_Q, item_id)

layer_1 = tf.layers.dense(inputs=tf.concat([mlp_item_latent_factor, mlp_user_latent_factor], axis=1),
                          units=num_factor_mlp * 2, kernel_initializer=tf.random_normal_initializer,
                          activation=tf.nn.relu,
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_rate))

layer_2 = tf.layers.dense(inputs=layer_1, units=hidden_dimension * 2, activation=tf.nn.relu,
                          kernel_initializer=tf.random_normal_initializer,
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_rate))

MLP = tf.layers.dense(inputs=layer_2, units=hidden_dimension, activation=tf.nn.relu,
                      kernel_initializer=tf.random_normal_initializer,
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_rate))

pred_y = tf.nn.sigmoid(tf.reduce_sum(MLP, axis=1))

cost=rate - average - b_u - b_i + pred_y

normalpath=tf.square(cost)                                        #�õ����������
regpath=beta * (tf.nn.l2_loss(Xu - Yi) + tf.nn.l2_loss(b_u) + tf.nn.l2_loss(b_i))            #�õ��������
loss=tf.reduce_sum(normalpath) +regpath                                      #�õ���ʧ����
trainer=tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)          #�Ż���
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
losses=[]                                                       #ÿ�θ��µ���ʧ�����б�
rmselist=[]
maelist=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(steps):
        train_loss_list=[]                                            #�������Ƭ����ʧ����
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