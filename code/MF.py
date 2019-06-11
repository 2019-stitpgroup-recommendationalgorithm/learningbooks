try:
    import numpy
except:
    print( "This implementation requires the numpy module.")
    exit(0)

def matrix_factorization(R, P, Q, K,test, steps=200, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):       #当0.001一直达不到强制退出
        for i in range(len(R)):     #行
            for j in range(len(R[i])):      #列
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])    #得到当前i*j的值
                    for k in range(K):          #更新i*j的预测值
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)         #求出整个曲线的矩阵方程
        e = 0
        for i in range(len(R)):      #梯度下降求偏差
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:       #当斜率小于0.001时，停止梯度下降，否则循环到最终次数
            break
        rmse=0
        mae=0
        for deep in range(len(test)):   #计算测试数据误差
            r=test[deep][0]
            l=test[deep][1]
            rate=test[deep][2]
            difference=numpy.dot(P[r,:],Q[:,l])-rate
            rmse+=numpy.square(difference)
            mae+=numpy.absolute(difference)
        rmse=numpy.sqrt(rmse/20000)
        mae/=20000
        print("RMSE值:",rmse)
        print("MAE值:",mae)
    return P, Q.T

################################################################
# ###############

if __name__ == "__main__":
    f=numpy.loadtxt('../TrainDate/ml-100k/u.data',int)
    f=numpy.delete(f,3,1)       #得到所有的有效信息
    R=numpy.zeros((943,1682))
    test=numpy.zeros((20000,3),int)
    delt=numpy.arange(100000)
    numpy.random.shuffle(delt)
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
    K = 2

    P = numpy.random.rand(N,K)  #随机初始化
    Q = numpy.random.rand(M,K)

    nP, nQ = matrix_factorization(R, P, Q, K,test)