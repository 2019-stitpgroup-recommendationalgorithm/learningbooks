[TOC]

# pandas简易学习笔记

## 读取文件

1. *pd.read_csv(src,sep,header,names)* ：
      1. *src* 代表 `路径`
      2. *sep* 代表 `分隔符` 
      3.  *header* 代表 `起始行` ，数据的第一行默认读取为列名(设置为None或者设置了names属性)
      4.  *names* 代表每一列代表的标题( `列名` )

## 数据处理

1. *pd.Series(list)* ：将list转化为键值对(json)形式

2. *pd.unique(values)* ：得到其中所有值不同的list
	  1. 等效于*values.unique()*
3. *pd.shape[纬度]* ：得到此纬度上的数据总数(行/列)
4. *pd.DataFrame(data,columns)* ：生成每行的列标题为columns的数据（类似矩阵，但是每列都有标题）
5. *pdDataFrame.itertuples* ：遍历行的迭代器(for in)
6. *dataframe名.标题名.replace(dict,inplace=)*：根据dict==替换==此部分的数据
     1. dict：字典或包含key、value的二元组的list
     2. inplace：是否替换源数据
7. 
