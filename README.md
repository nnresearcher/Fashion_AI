# Fashion_AI

## 执行依赖的环境和库

### 在训练模块中需要的库

  random
  
  math
  
  PIL
  
  keras
  
  numpy
  
  tensorflow
  
  os
  
  pandas
  
  
### 在测试模块中需要的库为


  PIL
  
  numpy
  
  os
  
  keras
  
  pandas
  
  
## 训练步骤说明
  
### 过渡文件的生成

运行 preprocess.py
  
  
1.导入训练数据base与标签label
    
    
2.图像的标签扩展为54位的标签，将原有的标签导入到该标签中设为1，其余为0
    
    
3.为避免过拟合现象，将训练数据划分出一部分为测试数据
     
     
4.将图像的路径与54位二进制标签对应保存在csv文件中。
    
    
### model模型的建立

  该函数为子函数，不需要运行。
  
  
1.导入densenet121的模型结构，该模型结构去除了原模型的结尾。
  
  
2.在densenet121模型结尾加上两层全连接层，最后一层使用sigmoid作为激活函数，其余使用relu作为激活函数。
  
  
3.在初始化模型时保留densenet121与本模型共享层的权重作为初始权重。
  

### 训练函数生成model

  运行run.py
  
  
1.导入help.py文件中的帮助函数
    
    
2.设置算法各个参数
    
    
3.生成训练数据与测试数据
    
    
4.导入初始的模型结构以及权重
    
    
5.使用SGD作为学习算法，为增加后期学习效果，使用训练函数model.fit_generator中的callback参数使学习率下降。
    
    
6.保存学习后的权重
    
  
  
## 测试步骤说明
