# CIFAR10_TEST_CYH

这个项目的目的是通过CIFAR10数据集学习[PyTorch](http://pytorch.org/)	。

本项目仿照[浙大Mo](https://mo.zju.edu.cn/)上的示例，基于CIFAR10数据集进行图片分类识别的模型实现和训练。

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

## 优化方案
- 数据增强：对原始32*32的图像四周各填充4个0像素（40*40），然后随机裁剪成32*32的新图像。根据测试可以将最后的准确率提高10%；
- 增大batch size：调节训练集上的batch size为128
- 调节学习率为0.001，相较一开始的0.01学习率，最终收敛的准确度更高。

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