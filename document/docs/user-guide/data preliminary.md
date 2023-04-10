# 数据表格智能分析和可视化

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-2.jpg?raw=true" , width="400px" />
</p>

该功能模块为特征工程提供初步的数据智能分析和可视化展示，主要实现特征数据和目标数据在数据集中的分布可视化，关系可视化。

## 数据表格智能分析

用户登录时，在`Data Preliminar`功能模块下，单击`Data Profiling`按钮。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-4.jpg?raw=true" , width="400px" />
</p>

进入`Data Profiling`模块，页面弹出如下图所示的`.csv`文件上传框。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-5.jpg?raw=true" , width="400px" />
</p>


<Upload File 功能>

上传数据之后，页面显示数据的智能分析报告。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-7.jpg?raw=true" , width="400px" />
</p>

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-8.jpg?raw=true" , width="400px" />
</p>

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-9.jpg?raw=true" , width="400px" />
</p>

## 数据变量关系可视化分析

用户登录时，在`Data Preliminar`功能模块下，单击`Data Visualization`按钮。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-10.jpg?raw=true" , width="400px" />
</p>

### 数据表格信息

进入`Data Profiling`模块，上传数据之后，`Data Table`功能显示加载所上传的`.csv`文件的数据，可通过调节`rows`调整显示的数据表的行数。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-11.jpg?raw=true" , width="400px" />
</p>

### 数据统计信息

`Data Statistics`功能显示所上传数据的统计信息，点击`download`可进行下载

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-12.jpg?raw=true" , width="400px" />
</p>

### 选择目标变量

`Features vs Targets`功能显示数据集的特征变量和目标变量，默认`.csv`文件中的最后一列为目标变量，可通过`input target`调节目标变量的个数。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-14.jpg?raw=true" , width="400px" />
</p>

### 特征变量分布

`Feature Statistics Distribution`功能显示每个特征变量数值分布统计直方图并给出核密度估计曲线，可通过`Plot parameters`功能调节图像的颜色、字体、标题和刻度大小。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-15.jpg?raw=true" , width="400px" />
</p>

### 目标变量分布

`Target Statistics Distribution`功能显示每个特征变量数值分布统计直方图并给出核密度估计曲线，可通过`Plot parameters`功能调节图像的颜色、字体、标题和刻度大小。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-16.jpg?raw=true" , width="400px" />
</p>

### 特征变量配方分布

`Feature Recipe Distribution`功能按照数据集中特征的顺序统计每个特征在样本中的数量，从而得知目标的常规配方，可通过`Plot parameters`功能调节图像的颜色、字体、标题和刻度大小。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-17.jpg?raw=true" , width="400px" />
</p>

### 特征变量数据集分布

`Distribution of Feature in Dataset`功能统计特征在数据集中的分布情况，可通过`Plot parameters`功能调节图像的颜色、字体、标题和刻度大小。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-18.jpg?raw=true" , width="400px" />
</p>

### 特征变量与目标变量

`Features and Targets`功能显示特征数据和目标数据的关系，可通过`Plot parameters`功能调节图像的颜色、字体、标题和刻度大小。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-19.jpg?raw=true" , width="400px" />
</p>

### 目标变量与目标变量

`Tagrets and Targets`功能显示特征数据和目标数据的关系，可通过`Plot parameters`功能调节图像的颜色、字体、标题和刻度大小 。

如果是多目标数据集，`Tagrets and Targets`功能显示目标数据和目标数据的关系，可通过`Plot parameters`功能调节图像的颜色、字体、标题和刻度大小 。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-20.jpg?raw=true" , width="400px" />
</p>


