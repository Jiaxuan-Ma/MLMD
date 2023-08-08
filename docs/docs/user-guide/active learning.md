# 主动学习模块 [![](https://img.shields.io/badge/Bgolearn-GitHub-green)](https://github.com/Bin-Cao/Bgolearn)

本模块通过调用开源算法库 **Bgolearn**  实现(📒[手册](https://bgolearn.netlify.app/))。用于材料成分定向设计以及性能定向优化过程。***通过已有实验数据（样本）及其测试性能，在给定的成分空间中搜索最优的材料成分设计，以将目标性能最大/最小化***。推荐的成分通过实验合成后，变成新的数据加入数据集合，Bgolearn将利用更多的数据信息对下一次设计做出更加可靠的推荐。迭代这个过程可以高效地在给定的成分空间中，寻找到具有优秀性能的新材料。***其中所有的实验过程也可以通过模拟过程代替***。如下图：

________________________________________________________________

<p align="center">
  <img src="https://user-images.githubusercontent.com/86995074/230322616-08fc629c-1858-42e7-8795-57fc8d076339.png" , width="400px" />
</p>


关于主动学习中虚拟空间采样点的说明：

1. 如果只上传一个数据文件，则文件默认为[标准数据集](https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/data%20layout.jpg)，虚拟采样点通过主动学习模块的`sample space ratio`和`sample number`生成，注意自动生成的虚拟采样点可能违背物理规律（例如，特征数据为合金的成分时，推荐的样本成分总和大于100 w.t%）

2. 如果上传两个文件，则第一个文件默认为[标准数据集](https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/data%20layout.jpg)，第二个文件默认为虚拟采样点文件。
虚拟采样点`.csv`文件数据布局如下：

________________________________________________________________

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231684019-26ee6d76-ae32-4386-a188-748fb78a34bc.jpg?raw=true" , width="400px" />
</p>

**操作**

在`Home`主页，进入`active learning`模块

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231684031-85ceb45c-c11f-4cb7-be13-ded376b743eb.jpg?raw=true" , width="400px" />
</p>

页面弹出如下图所示的`.csv`文件上传框。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231178930-06bb0b95-1765-46bc-8011-d4932c7d7ea1.jpg?raw=true" , width="400px" />
</p>


***Active Learning*** 模块 [![](https://img.shields.io/badge/Bgolearn-GitHub-green)](https://github.com/Bin-Cao/Bgolearn)  : 

上传数据之后，`Data Table`功能显示加载所上传的`.csv`文件的数据，通过调节`rows`调整显示的数据表的行数。

``Features vs Targets``功能显示数据集的特征变量和目标变量，默认`.csv`文件中的最后一列为目标变量，可通过`input target`调节目标变量的个数。


+ `Choose Target`-选择目标特征

+ 在`Sampling`功能下选择`model`，在`Hyper Parameters`中可调节算法的超参数

Note : 超参数说明, 见 Bgolearn 📒[手册](https://bgolearn.netlify.app/recommend-r/)
________________________________________________________________

+ `simple number` 选择推荐的样本个数

+ `min search` 选择优化的方向（最小/最大）

+ `sample criterion` 选择bayes推荐算法, 共包括九种效用函数

> 1: `Expected Improvement algorith`-期望提升函数
> 
> 2:`Expected improvement with "plugin"`-有“plugin”的期望提升函数
> 
> 3:`Augmented Expected Improvement`-增广期望提升函数
> 
> 4:`Expected Quantile Improvement`-期望分位提升函数
> 
> 5:`Reinterpolation Expected Improvement`-重插值期望提升函数
> 
> 6:`Upper confidence bound`-高斯上确界函数
> 
> 7:`Probability of Improvement`-概率提升函数
> 
> 8:`Predictive Entropy Search`-预测熵搜索函数
> 
> 9:`Knowledge Gradient`-知识梯度函数

________________________________________________________________

+ 在`sample space ratio`中选择虚拟空间采样点范围:

$$
virtual_{sapce} = [ ratio_{min} \times X_{min},  ratio_{max} \times X_{max} ]
$$

+ `sample numeber`选择每个特征的采样个数

根据所选择的算法和超参数推荐的样本可点击`download`可下载。



