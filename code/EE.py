import numpy as np

def EE(R,average,b1,b2,X,Y,steps=5000,alpha=0.001,beta=0.02):       #alpha是梯度下降的步长，beta是正则化项系数
    cost=float("inf")
    for step in range(steps):       #总次数
        for i in range(len(R)):                     #更新b1，b2，x，y的值
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    p=X[i]-Y[j]
                    eij = R[i][j] - average - b1[0][i] - b2[0][j] + np.dot(p,p.T)    #得到当前预测值-真实值
                    b1[0][i]=b1[0][i] + alpha * (eij - beta * (b1[0][i]))                     #更新b1的预测值
                    b2[0][j]=b2[0][j] + alpha * (eij - beta * (b2[0][j]))                     #更新b2的预测值
                    X[i] = X[i] - alpha * (eij + beta) * (p)            #更新X的预测值
                    Y[j] = Y[j] + alpha * (eij + beta) * (p)            #更新Y的预测值
        nowcost=0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j]>0:
                    p=X[i]-Y[j]
                    nowcost += np.square(R[i][j] - average-b1[0][i]-b2[0][j]+np.dot(p,p.T))
                    #开始计算正则化项
                    p=pow(b1[0][i],2)+pow(b2[0][j],2)           #b1^2+b2^
                    for k in range(len(X[0])):                  #(x-y)^2
                        p+=pow(X[i][k]-Y[j][k],2)
                    nowcost+=alpha*p
        rmse=0
        mae=0
        for deep in range(len(test)):   #计算测试数据误差
            r=test[deep][0]
            l=test[deep][1]
            rate=test[deep][2]
            p=X[r,:]-Y[l,:]
            difference=rate - average-b1[0,r]-b2[0,l]+np.dot(p,p.T)
            rmse+=np.square(difference)
            mae+=np.absolute(difference)
        rmse=np.sqrt(rmse/20000)
        mae/=20000
        print("RMSE值:",rmse)
        print("MAE值:",mae)
        if cost-nowcost < 0.001:               #偏差很小则停止
            break
        cost=nowcost
    return
if __name__ == '__main__':
    f=np.loadtxt('../TrainDate/ml-100k/u.data',int)
    f=np.delete(f,3,1)       #得到所有的有效信息
    average=np.average(f[:,2])      #计算评分的平均值
    R=np.zeros((943,1682))
    test=np.zeros((20000,3),int)
    delt=np.arange(100000)
    np.random.shuffle(delt)
    delt=delt.reshape(5,20000)[0]   #随机得到20%的测试数据
    delt.sort()
    flag=0
    for rows in range(len(f)):
        if flag ==20000 or rows!=delt[flag]:
            R[f[rows][0]-1][f[rows][1]-1]=f[rows][2]        #得到训练数组
        else:
            test[flag][0]=f[rows][0]-1
            test[flag][1]=f[rows][1]-1
            test[flag][2]=f[rows][2]
            flag+=1      #得到测试数组
    N = len(R)
    M = len(R[0])
    K=5
    b1=np.random.rand(1,N)
    b2=np.random.rand(1,M)
    X = np.random.rand(N,K)  #随机初始化
    Y = np.random.rand(M,K)
    EE(R,average,b1,b2,X,Y)