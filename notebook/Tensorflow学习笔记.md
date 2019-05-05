[TOC]

# Tensorflow学习笔记

## tensorflow的总认识

1. tensor是张量的意思，在tensorflow中作为数据和变量
2. tensorflow中定义的变量不会直接初始化，需要调用初始化函数并运行才会有值

## tensorflow的大体流程

注：我们将tensorflow起名为tf

1. 定义变量：tf.Variable()
2. 定义常量：tf.Constant()
3. 矩阵乘法：tf.matmul()
4. 矩阵中每个元素平方：tf.square()
5. 初始化变量：tf.global_variables_initializer()

  1. ps：tf.initialize_all_variables()将不再使用
6. 计算某一纬度上的平均值：tf.reduce_mean(需要计算的tensor)
7. 选择梯度下降模型：tf.train.GradientDescentOptimizer(梯度下降步长)
8. 启动图 (graph)：
  1. 定义session上下文：sess = tf.Session()
  2. 运行程序并且返回需要输出的值：sess.run(需要输出的值)

## python知识的补充学习

### 基础

1. python中代码缩进就相当于其他语言的花括号
2. for、if等语句行末加：表示结束，进行循环体
3. for in循环：
	1. for 变量 in range(数字)：进行迭代
4. 导库：
	1. import xxx：直接导入整个库，通过xxx.方法调用
	2. import xxx as aaa： 将xxx库整体导入，并且起名为aaa
	3. from xxx import aaa：导入xxx库中的aaa函数/类等，不整体导入xxx库

### 方便使用

1. with xxx as aaa：给xxx起名为aaa，可以使用aaa直接调用xxx，行末需要加：后面的代码需要缩进
2. 

