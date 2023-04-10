# 特征工程模块

## 特征变量缺失值处理

`Feature Engineering`-`Missing Features `模块

### 丢弃特征变量缺失值

单击`Drop Missing Fearures`按钮，上传`.csv`文件之后，可以在`Drop Missing Features`功能下拉动`Missing Threshold`进度条，选择丢弃的缺失值特征的阈值，点击`download`可下载处理之后的数据。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-21.jpg?raw=true" , width="400px" />
</p>

### 填补特征变量缺失值

单击`Fill Missing Features`按钮，上传`.csv`文件之后，可以在`Fill Missing Features`功能下进行缺失值数据填补。`fill method` 选择填补方法，`missing feature`选择填补的特征，可以选择多个特征。
`fill method`-`fill in normal method`中可以选择`mean, constant, median, most frequent`特征均值、常数（默认为0），中位数和众数四种填补方式。

`fill method`-`fill in RandomForestRegression`中使用随机森林算法进行所有特征的空缺值填补，其中`mean, constant, median, most frequent`表示随机森林训练时填补特征的方式。
点击`download`可下载处理之后的数据。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-22.jpg?raw=true" , width="400px" />
</p>

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-23.jpg?raw=true" , width="400px" />
</p>

### 丢弃特征变量中的唯一值

在`Feature Engineering`- `Drop Nunique Features `模块下:

单击`Drop Nuniqe Fearures`按钮，上传`.csv`文件之后，在`Drop Nunqiue Features`功能下拉动`drop unique counts`进度条，选择丢弃的数值唯一性的特征的阈值，`count=1`代表丢弃数值在所有样本中都相同的特征，`count=2`代表丢弃数值在所有样本中只有两个值的特征，依次类推`count=3...`，在`drop unique counts`进度条下方的`nunqiue `表格中显示特征唯一值的统计数量。右侧表格显示处理之后的数据，点击`download`可下载。
`Plot`扩展栏中绘制了特征数据唯一值数量统计直方图，可调节图像的颜色、字体、标题和刻度大小

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-24.jpg?raw=true" , width="400px" />
</p>

### 特征变量与目标变量相关性

在`Feature Engineering`- `Correlation of Features vs Targets `模块下:
点击`Drop Low Correlation Features vs Target`按钮， 上传`.csv`文件之后，在`Drop Low Correlation Features vs Target `功能下`choose target`选择目标变量，显示特征与所选择目标的相关性横向直方图。`correlation method`中选择相关性方法中选择`pearson,spearman,kendall,MIR` 皮尔森相关性系数、斯皮尔曼相关性系数、肯德尔相关性系数（类别变量）、互信息方法。`corr thershold f_t`进度条中选择特征数据和目标的相关性阈值，低于阈值的特征将被丢弃。`Processed Data`中可点击`download `下载处理之后的数据。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-25.jpg?raw=true" , width="400px" />
</p>

### 特征变量与目标变量相关性

在`Feature Engineering`- `Correlation of Features vs Features `模块下:

点击`Drop Collinear Features`按钮， 上传`.csv`文件之后，在`Drop Collinear Features`功能下`choose target`选择目标变量，显示特征与所选择目标的相关性系数热力图。`correlation method`中选择相关性方法中选择`pearson,spearman,kendall,` 皮尔森相关性系数、斯皮尔曼相关性系数、肯德尔相关性系数（类别变量）。在`correlation threshold`进度条中选择特征数据和特征数据之间的相关性阈值，高于阈值的两个特征将被筛选出来，丢弃其中与目标相关性更低的特征。在`Processed Data`中可点击`download `下载处理之后的数据。`is mask`功能选择是否将热力图进行掩码展示。`drop features`中显示丢弃的特征。`Processed Data`中可点击`download `下载处理之后的数据。

### 类别特征变量one-hot编码

在`Feature Engineering`- `One-hot Encoding Features `模块下:
点击`One-hot Encoding `按钮，上传`.csv`文件之后，在`One-hot encoding Features`中将会显示one-hot编码之后的数据，如类别特征`Sex`中值`female`和`male`将被转换为`0,1`和`1,0`，并删除旧特征`Sex`，创建新特征`Sex_female`和`Sex_male`添加到数据集中。`Processed Data`中可点击`download `下载处理之后的数据。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-26.jpg?raw=true" , width="400px" />
</p>

### 特征变量重要性排序

在`Feature Engineering`- `Features Importance `模块下:
点击`Feature Importance`按钮，上传`.csv`文件之后，在`Choose Target`功能下选择目标特征。在`Selector`功能下选择`model`，其中`RandomForestClassifier`负责分类目标数据的特征重要性排序。`LassoRegressor, LinearRegressor,RandomForestRegressor, RidgeRegressor`负责连续目标数据的特征重要性排序。`Hyper Parameters`中选择不同算法的超参数，`cumulative importance `选择按照特征重要性从大到小排列加和的阈值，舍弃阈值之后的特征。点击`Embedded method`将使用嵌入法按照特征从到小的顺序依次添加训练模型，可视化不同重要性的特征对模型的影响，`cv`可选择交叉验证的折数。
点击`train`按钮，根据所选择的算法和超参数进行特征重要性排序，给出特征重要性计算表格，并绘制特征重要性直方图。`Processed Data`中可下载经过`dropped zero importance `的数据和经过`dropped low importance`的数据。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-27.jpg?raw=true" , width="400px" />
</p>

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-28.jpg?raw=true" , width="400px" />
</p>

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-29.jpg?raw=true" , width="400px" />
</p>

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-30.jpg?raw=true" , width="400px" />
</p>

