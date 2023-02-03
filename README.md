# image-classification
这是我最近学习的图像处理的代码，我自己进行了整合，修改  
目前网络包含：  
`lenet`,`alexnet`,`googlenet`,`vgg`,`resnet`,`mobilenet`,`shufflenet`,`senet`,`efficient`  
持续更新中。。。。。
## 训练权重
| 网络 | 权值文件名称 | 下载链接 | 输入图片大小 | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| resnet18 | [resnet18-5c106cde.pth](https://download.pytorch.org/models/resnet18-5c106cde.pth) | https://download.pytorch.org/models/resnet18-5c106cde.pth | 640x640 |111  | 46
| resnet34 | [resnet34-333f7ec4.pth](https://download.pytorch.org/models/resnet34-333f7ec4.pth) | https://download.pytorch.org/models/resnet34-333f7ec4.pth | 640x640 | 27.4 | 44.5
| resnet50 | [resnet50-19c8e357.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth) | https://download.pytorch.org/models/resnet50-19c8e357.pth | 640x640 | 34.7 | 53.6
```
权重更新中
```

## 训练方法
 **两种训练方法可选**
1. 在`train.py`里面修改网络超参数，直接运行`train.py`文件
2. 在终端输入如下命令：
```python
python train.py --batchsize=8 --epochs=10 --lr=0.0001
```


## 预测
提供两种预测方法，单独预测和批量预测  
  1.`predict.py`
  2.`batch_predict.py`













## Reference
https://github.com/WZMIAOMIAO
