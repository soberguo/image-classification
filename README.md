# image-classification
这是我最近学习的图像处理的代码，我自己进行了整合，修改
## 训练权重
```python
* resnet
  * resnet34
      * https://download.pytorch.org/models/resnet34-333f7ec4.pth
  * resnet50
      * https://download.pytorch.org/models/resnet50-19c8e357.pth
```
| 网络 | 权值文件名称 | 测试数据集 | 输入图片大小 | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| resnet34 | [resnet34-333f7ec4.pth](https://download.pytorch.org/models/resnet34-333f7ec4.pth) | COCO-Val2017 | 640x640 | 27.4 | 44.5
| resnet50 | [resnet50-19c8e357.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth) | COCO-Val2017 | 640x640 | 34.7 | 53.6
| COCO-Train2017 | [yolox_s.pth](https://github.com/bubbliiiing/yolox-pytorch/releases/download/v1.0/yolox_s.pth) | COCO-Val2017 | 640x640 | 38.2 | 57.7
| COCO-Train2017 | [yolox_m.pth](https://github.com/bubbliiiing/yolox-pytorch/releases/download/v1.0/yolox_m.pth) | COCO-Val2017 | 640x640 | 44.8 | 63.9
| COCO-Train2017 | [yolox_l.pth](https://github.com/bubbliiiing/yolox-pytorch/releases/download/v1.0/yolox_l.pth) | COCO-Val2017 | 640x640 | 47.9 | 66.6
| COCO-Train2017 | [yolox_x.pth](https://github.com/bubbliiiing/yolox-pytorch/releases/download/v1.0/yolox_x.pth) | COCO-Val2017 | 640x640 | 49.0 | 67.7














## Reference
https://github.com/WZMIAOMIAO
