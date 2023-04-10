# 主动学习模块

关于主动学习中虚拟空间采样点的说明：

1. 如果只上传一个数据文件，则文件默认为[标准数据集](https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/data%20layout.jpg)，虚拟采样点通过主动学习模块的`sample space ratio`和`sample number`生成，注意自动生成的虚拟采样点可能违背物理规律（例如，特征数据为合金的成分时，推荐的样本成分总和大于100 w.t%）

2. 如果上传两个文件，则第一个文件默认为[标准数据集](https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/data%20layout.jpg)，第二个文件默认为虚拟采样点文件。
虚拟采样点`.csv`文件数据布局如下：

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/activate%20learning/Mjx-20230409-8.jpg?raw=true" , width="400px" />
</p>


在`Home`主页，进入`active learning`模块

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/activate%20learning/Mjx-20230409-9.jpg?raw=true" , width="400px" />
</p>

页面弹出如下图所示的`.csv`文件上传框。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-5.jpg?raw=true" , width="400px" />
</p>


`Regression`- `Data Information `模块:

上传数据之后，`Data Table`功能显示加载所上传的`.csv`文件的数据，可通过调节`rows`调整显示的数据表的行数。

``Features vs Targets``功能显示数据集的特征变量和目标变量，默认`.csv`文件中的最后一列为目标变量，可通过`input target`调节目标变量的个数。


`Choose Target`功能选择目标特征

在`Sampling`功能下选择`model`，在`Hyper Parameters`中可调节算法的超参数

`simple number` 选择推荐的样本个数

`min search` 选择优化的方向（最小/最大）

`sample criterion` 选择bayes推荐算法

> `Expected Improvement algorith`
> 
> `Expected improvement with "plugin"`
> 
> `Augmented Expected Improvement`
> 
> `Expected Quantile Improvement`
> 
> `Reinterpolation Expected Improvement`
> 
> `Upper confidence bound`
> 
> `Probability of Improvement`
> 
> `Predictive Entropy Search`
> 
> `Knowledge Gradient`



在`sample space ratio`中选择虚拟空间采样点范围:

$$
virtual_{sapce} = [ ratio_{min} \times X_{min},  ratio_{max} \times X_{max} ]
$$

`sample numeber`选择每个特征的采样个数

根据所选择的算法和超参数推荐的样本可点击`download`可下载。

