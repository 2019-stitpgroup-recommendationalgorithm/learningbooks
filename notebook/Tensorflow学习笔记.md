[TOC]

# Tensorflow学习笔记

## tensorflow的总认识

1. tensor是张量的意思，在tensorflow中作为数据和变量
2. tensorflow中定义的变量不会直接初始化，需要调用初始化函数并运行才会有值
3. 张量(tensor)的阶数即为数组的维数，二阶的张量就是平时所说的矩阵

## tensorflow的大体流程

注：我们将tensorflow起名为tf

### 总体基础

#### 声明和输出(流程部分)

1. 声明tensor：
     1. 按照所给的数值和形式生成对应形状的tensor***变量***：*tf.Variable(values)* 
     2. 按照所给的数值和形式生成对应形状的tensor***常量***：*tf.constant(初始化参数)*
     3. 占位符：*tf.placeholder(dtype,shape)* ：先占用一个位置，不初始化，调用的时候再赋值
          1. 调用占位符：运行表达式的时候给 *feed_dict* 参数赋值
2. 初始化变量：
     1. *tf.global_variables_initializer()* ====> 初始化所有变量
          1. ps：~~*tf.initialize_all_variables()*~~方法已废弃
     2. *变量名.initializer.run()* ===> 初始化单个变量
          1. 需要使用*tf.InteractiveSession()* 或 with session的.as_default() 方法
3. 计算某一纬度上的平均值：*tf.reduce_mean(需要计算的tensor)*
4. 选择梯度下降模型(梯度下降优化器)：*tf.train.AdamOptimizer(梯度下降步长)*
     1. `.minimize(x)` ：在梯度下降过程中最小化x的值
5. 启动图 (graph)：
     1. 定义session上下文：*sess = tf.Session()*
       1. *tf.InteractiveSession()*：定义交互式session，可以通过变量调用*run()*方法和*eval()*方法而不需要通过session
       2. 运行程序并且返回需要取得的值：*sess.run(需要取的值)*
            1. eval表达式：*表达式.eval()* <=\=等同于\==> *sess.run(表达式)* 
       3. *sess.close()*：释放资源

#### 矩阵生成

1. *tf.ones(shape,type)*：生成全1矩阵
2. *tf.zeros(shape,type)*：生成0矩阵
3. *tf.fill(shape,value)*：根据shape创建一个值全为value的矩阵
4. *tf.identity(x)* ：生成长宽为x的单位矩阵(对角线全1)
5. *tf.Variable(values)* ：按照所给的数值和形式生成对应形状的tensor变量
6. *tf.constant(values)* ：按照所给的数值和形式生成对应形状的静态tensor

##### 高级

1. *tf.random_normal(shape,mean,stddev)* ：生成***正态分布***的随机矩阵，mean为均值，stddev为标准差
2. *tf.truncated_normal(shape,mean,stddev)* ：生成***截断正态分布***的随机矩阵，值同上
	1. 截断标准是两倍stddev，即取值[-2stddev,2stddev]
	2. 可用来给标准正态分布添加噪音
3. *tf.random_uniform(shape,minval,maxval)* ：生成***均匀分布***的随机矩阵，范围为[minval,maxval]

#### 运算

1. 加法：
     1. *tf.add(x,y)* ===> 得到`x+y`
          1. 纬度不一样则每行都会添加
     2. *tf.reduce_sum(x,axis,keepdims)* ：计算某个纬度上的和，axis代表纬度(不写代表全部求和)，keepdims为是否保持原纬度输出
2. 减法：*tf.subtract(x,y)* ===> 得到`x-y`
3. 乘法：
     1. 矩阵乘法：*tf.matmul(a,b)* ===> 得到`a*b`
       2. 每个元素平方：*tf.square(x)* ===> 得到`x^2`
       3. 每个元素对应相乘：*tf.multiply(x,y)* ===> 得到x每个元素和y每个元素相乘的矩阵
       4. 标量(常数)和张量(tensor)相乘：*tf.scalar_mul(标量,张量)*
4. 除法：
     1. 矩阵除法：~~*tf.div(x,y)*~~ ===> 得到`x/y`
     	   1. 如果其中一个是浮点数，则结果是浮点型，否则是整型
     	   2. Python 2.7语法，不支持*tf.math.divide*，已废弃
       2. 浮点数除法：*tf.divide(x, y)* ===> 替代div方法，得到`x/y`
5. 计算某纬度上的平均值： *tf.reduce_mean(x,axis,keepdims)* 使用方法同*reduce_sum*
6. 取余：*tf.mod(x, y, name=None)* ===> 得到`x%y`
7. 开根：*tf.sqrt(x)*
8. 取负：*tf.negative(x)* ===> 得到`-x`
9. 向上取整：*tf.ceil(x)*
10. 向下取整：*tf.floor(x)*
11. 返回其中的大值：*tf.maximum(x, y)*
12. 返回其中的小值：*tf.minimum(x, y)*

#### 其他操作

##### 信息

1. *tf.shape(tensor)*：返回tensor的形状

##### 更改和生成

1. *tf.pack(values,axis)* ：将values中的张量按照axis轴进行合并，合并成一个新的张量
     1. values是个数组，每一项为需要合并的张量
       2. axis为数字，代表需要合并的纬度(轴)
2. *tf.concat(concat_dim,values)* ：作用同上，只是第1、2个参数互换
3. *tf.where(条件)* ：返回符合条件的索引下标的tensor
4. *tf.nn.embedding_lookup(x,y)* ：从x中取出y下标的部分，返回tensor，可以`同时处理多个tensor，多个纬度`
5. *tf.gather(a,b)* ：从a中取出和b中的索引相同的，并返回tensor，***本质同上***，但是一次只能处理一个tensor
6. *tf.gather_nd(a,b)* ：***同上***，可以使用多纬，一次只能处理一个tensor
7. *tf.nn.relu(a)* ：将a中小于0的全部变为0，其余不变，返回一个新的tensor

##### 判断

1. 判断相等：*tf.equal(x,y)* ===>判断每个元素是否相等，返回bool格式的tensor	

##### 转化

1. *tf.cast(x,dtype)* ：将tensor x转化成新数据类型的tensor
2. *tf.transpose(a)* ：将矩阵a进行转置(只有二维矩阵能转置，否则报错)

##### 求值

1. *tf.gradients* ：求导数

##### 更新

1. *tf.assign(prevalue,nextvalue)* ：更新其中变量的值，将nextvalue赋值给prevalue

#### 优化器( `tf.train` )

注：更多更详细的请查询 <https://zhuanlan.zhihu.com/p/34169434> 

1. `.AdagradOptimizer` ：随机梯度下降，先快后慢
2. `.AdamOptimizer` ：稳定步长的梯度下降
3. 

### 深度学习

#### mlp层

1. *tf.layers.dense(inputs, units, activation)*：将各层进行连接
	1. input：输入tensor
	2. units：输出神经节点的节点数
	3. activation：激活函数
2. *tf.nn.dropout(x,keep_prob=,rate=)* ：
	1. keep_prob：



## python知识的补充学习

### 基础

1. python中代码缩进就相当于其他语言的花括号

2. for、if等语句行末加：表示结束，进行循环体

3. for in循环：

  4. for 变量 in range(数字)：进行迭代

5. 导库：
	  1. import xxx：直接导入整个库，通过xxx.方法调用
	  2. import xxx as aaa： 将xxx库整体导入，并且起名为aaa
	  3. from xxx import aaa：导入xxx库中的aaa函数/类等，不整体导入xxx库

6. main函数写法:(可以没有)

  ```python
  if __name__ == "__main__":
  ```

### 方便使用

1. with xxx as aaa：给xxx起名为aaa，可以使用aaa直接调用xxx，行末需要加：后面的代码需要缩进
	1. 作用：结束之后自动关闭

### 类

1. *class 类名(父类):* 定义一个类，没有父类可不写括号
2. 类中需要定义一个*def \_\_init\_\_(self,参数):*的初始化函数
	1. *self* 代表类的实例，即new出来的对象，即其他语言中的this指针
	2. *self* 可以不起这个名字，但是第一位就是 *self*
	3. 注： *\_\_init\_\_* 函数不是构造函数，构造函数是 *\_\_new\_\_* 函数，默认继承了object的new
3. 实例化类使用 *类名(初始化函数参数)* 和调用方法一样

