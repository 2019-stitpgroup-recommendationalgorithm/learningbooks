[TOC]

# Tensorflow学习笔记

## tensorflow的总认识

1. tensor是张量的意思，在tensorflow中作为数据和变量
2. tensorflow中定义的变量不会直接初始化，需要调用初始化函数并运行才会有值
3. 张量(tensor)的阶数即为数组的维数，二阶的张量就是平时所说的矩阵

## tensorflow的大体流程

注：我们将tensorflow起名为tf

### 总体基础

#### 声明和输出

1. 定义变量：*tf.Variable(初始化参数)*
2. 定义常量：*tf.constant(初始化参数)*
3. 初始化变量：
     1. *tf.global_variables_initializer()* ====> 初始化所有变量
          1. ps：<del>*tf.initialize_all_variables()*</del>方法已废弃
     2. *变量名.initializer.run()* ===> 初始化单个变量
          1. 需要使用*tf.InteractiveSession()* 或 with session的.as_default() 方法
4. 计算某一纬度上的平均值：*tf.reduce_mean(需要计算的tensor)*
5. 选择梯度下降模型：*tf.train.GradientDescentOptimizer(梯度下降步长)*
6. 启动图 (graph)：
     1. 定义session上下文：*sess = tf.Session()*
       1. *tf.InteractiveSession()*：定义交互式session，可以通过变量调用*run()*方法和*eval()*方法而不需要通过session
       2. 运行程序并且返回需要取得的值：*sess.run(需要取的值)*
            1. eval表达式：*表达式.eval()* <=\=等同于\==> *sess.run(表达式)* 
       3. *sess.close()*：释放资源

#### 矩阵操作

1. tf.ones(shape,type)：生成单位矩阵
2. tf.zeros(shape,type)：生成0矩阵
3. tf.

#### 运算

1. 矩阵加法：*tf.add(x,y)* ===> 得到`x+y`
2. 矩阵减法：*tf.subtract(x,y)* ===> 得到`x-y`
3. 乘法：
	1. 矩阵乘法：*tf.matmul(a,b)* ===> 得到`a*b`
	2. 矩阵中每个元素平方：*tf.square(x)* ===> 得到`x^2`
	3. 每个元素对应相乘：*tf.multiply(x,y)* ===> 得到x每个元素和y每个元素相乘的矩阵
	4. 标量(常数)和张量(tensor)相乘：*tf.scalar_mul(标量,张量)*
4. 除法：
	1. 矩阵除法：<del>*tf.div(x,y)*</del> ===> 得到`x/y`
		1. 如果其中一个是浮点数，则结果是浮点型，否则是整型
		2. Python 2.7语法，不支持*tf.math.divide*，已废弃
	2. 浮点数除法：*tf.divide(x, y)* ===> 替代div方法，得到`x/y`
5. 取余：*tf.mod(x, y, name=None)* ===> 得到`x%y`
6. 开根：*tf.sqrt(x)*
7. 取负：*tf.negative(x)* ===> 得到`-x`
8. 向上取整：*tf.ceil(x)*
9. 向下取整：*tf.floor(x)*
10. 返回其中的大值：*tf.maximum(x, y)*
11. 返回其中的小值：*tf.minimum(x, y)*

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

5. main函数写法:(可以没有)

	```python
	if __name__ == "__main__":
	```

### 方便使用

1. with xxx as aaa：给xxx起名为aaa，可以使用aaa直接调用xxx，行末需要加：后面的代码需要缩进

### 类

1. *class 类名(父类):*定义一个类，没有父类可不写括号
2. 类中需要定义一个*def \_\_init\_\_(self,参数):*的初始化函数
	1. *self* 代表类的实例，即new出来的对象
	2. *self* 可以不起这个名字，但是第一位就是 *self*
	3. 注： *\_\_init\_\_* 函数不是构造函数，构造函数是 *\_\_new\_\_* 函数，默认继承了object的new
3. 实例化类使用 *类名(初始化函数参数)* 和调用方法一样

