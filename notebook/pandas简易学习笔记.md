[TOC]

# pandas简易学习笔记

## 读取文件

1. *pd.read_csv(src,sep,header,names)* ：
  2. *src* 代表 `路径` 
  3. *sep* 代表 `分隔符` 
  4. *header* 代表 `起始行` ，数据的第一行默认读取为列名(设置为None或者设置了names属性)
  5. *names* 代表每一列代表的标题( `列名` )
6. *pd.Series(list)* ：将list转化为键值对(json)形式
7. *pd.unique(values)* ：得到其中所有值不同的list
	  1. 等效于*values.unique()*
8. *pd.shape[纬度]* ：得到此纬度上的数据总数(行/列)
9. *pd.DataFrame(data,columns)* ：生成每行的列标题为columns的数据（类似矩阵，但是每列都有标题）
10. *pdDataFrame.itertuples* ：遍历行的迭代器(for in)
