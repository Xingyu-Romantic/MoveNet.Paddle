# MoveNet

Google提供的在线演示：[https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=movenet](https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=movenet)

MoveNet 是一个 Bottom-up estimation model， 使用heatmap。

## 网络架构
主要分为三个部分：Backbone、Header、PostProcess
- Backbone：Mobilenetv2 + FPN
- Header：输入为Backbone的特征图，经过各自的卷积，输出各自维度的特征图。共有四个Header：分别为Center、KeypointRegression、KeypointHeatmap、Local Offsets
	- Center：[N, 1, H, W], 这里1代表当前图像上所有人中心点的Heatmap，可以理解为关键点，只有一个，所以通道为1。提取中心点两种方式：
		- 一个人所有关键点的算术平均数。
		- 所有关键点最大外接矩形的中心点。（效果更好）
	- KeypointHeatmap：[N, K, H, W]  N：Batchsize、K：关键点数量，比如17。H、W：对应特征图的大小，这里输入为$192 \times 192$ , 降采样四倍就是$48\times 48$ 。代表当前图像上所有人的关键点的Heatmap
	- KeypointRegresssion：[N, 2K, H, W]  K个关键点，坐标用$x, y$表示，那么就有2K个数据。这里$x, y$ 代表的是同一个人的关键点对于中心点的偏移值。原始MoveNet用的是特征图下的绝对偏移值，换成相对值（除以48转换到0-1），可以加快收敛。
	- LocalOffsets：[N, 2K, H, W] 对应K个关键点的坐标，这里是Offset，模型降采样特征图可能存在量化误差，比如192分辨率下x = 0 和 x= 3映射到48分辨率的特征图时坐标都变为了0；同时还有回归误差。

## 损失函数
KeypointHeadmap 和 Center 采用加权MSE，平衡了正负样本。
KeypointRegression 和LocalOffsets 采用了 L1 Loss。
最终各个Loss权重设置为1:1:1:1
```python
loss = paddle.pow((pre-target),2) weight_mask = target*k+1
paddle.pow(torch.abs(target-pre), 2) loss = loss*weight_mask
```



## 参考文献
1. [2021轻量级人体姿态估计模型修炼之路（附谷歌MoveNet复现经验） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/413313925)
2. [fire717/movenet.pytorch: A Pytorch implementation of MoveNet from Google. Include training code and pre-train model. (github.com)](https://github.com/fire717/movenet.pytorch)
3. https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=movenet

