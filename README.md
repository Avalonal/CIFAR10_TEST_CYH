# CIFAR10_TEST_CYH

这个项目的目的是通过CIFAR10数据集学习[PyTorch](http://pytorch.org/)	。

## 相关环境
- Python 3.8.3
- PyTorch 1.5.1

## 使用方式
```
# 训练：
python train.py

# 测试：
python test.py

```

## 准确率
最终当平均loss下降到0.55时，模型在CIFAR10测试集上的准确率为80%

## 相关文件
```
# config.py：
项目中公用的配置信息

# datasets.py
加载数据集到dataloader

# Net.net
模型网络

# test.py
使用测试集进行测试

# train.py
在训练集上训练模型

# model/model_00.pkl
模型文件

```