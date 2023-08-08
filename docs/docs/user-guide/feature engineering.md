# 特征工程


<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231189726-148e0dc9-9655-4fb1-8527-cdae428c4b3a.jpg
?raw=true" , width="400px" />
</p>

---

## 特征变量缺失值处理


**Feature Engineering**模块-`Missing Features `

### 丢弃特征变量缺失值

单击`Drop Missing Fearures`按钮，上传`.csv`文件之后，可以在`Drop Missing Features`功能下拉动`Missing Threshold`进度条，选择丢弃的缺失值特征的阈值，点击`download`可下载处理之后的数据。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231181931-19ba1f42-f2ec-4abe-9d7b-9d9e101f915b.jpg?raw=true" , width="400px" />
</p>



### 填补特征变量缺失值

单击`Fill Missing Features`按钮，上传`.csv`文件之后，可以在`Fill Missing Features`功能下进行缺失值数据填补。`fill method` 选择填补方法，`missing feature`选择填补的特征，可以选择多个特征。
`fill method`-`fill in normal method`中可以选择`mean, constant, median, most frequent`特征均值、常数（默认为0），中位数和众数四种填补方式。

`fill method`-`fill in RandomForestRegression`中使用随机森林算法进行所有特征的空缺值填补，其中`mean, constant, median, most frequent`表示随机森林训练时填补特征的方式。
点击`download`可下载处理之后的数据。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231181946-16aaf6e1-ca86-4b06-806e-142645b0e5cd.jpg?raw=true" , width="400px" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231181956-fcd93d65-338d-46e4-a37f-b64075d78bd8.jpg?raw=true" , width="400px" />
</p>

## 特征变量唯一值处理

---

在**Feature Engineering**模块- `Drop Nunique Features `模块下:

单击`Drop Nuniqe Fearures`按钮，上传`.csv`文件之后，在`Drop Nunqiue Features`功能下拉动`drop unique counts`进度条，选择丢弃的数值唯一性的特征的阈值，`count=1`代表丢弃数值在所有样本中都相同的特征，`count=2`代表丢弃数值在所有样本中只有两个值的特征，依次类推`count=3...`，在`drop unique counts`进度条下方的`nunqiue `表格中显示特征唯一值的统计数量。右侧表格显示处理之后的数据，点击`download`可下载。
`Plot`扩展栏中绘制了特征数据唯一值数量统计直方图，可调节图像的颜色、字体、标题和刻度大小

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231192789-6751c135-b6c2-4a86-b86d-08103579ee65.jpg?raw=true" , width="400px" />
</p>


## 特征变量与目标变量相关性

---

在**Feature Engineering**模块- `Correlation of Features vs Targets `模块下:
点击`Drop Low Correlation Features vs Target`按钮， 上传`.csv`文件之后，在`Drop Low Correlation Features vs Target `功能下`choose target`选择目标变量，显示特征与所选择目标的相关性横向直方图。`correlation method`中选择相关性方法中选择`pearson,spearman,kendall,MIR` 皮尔森相关性系数、斯皮尔曼相关性系数、肯德尔相关性系数（类别变量）、互信息方法。`corr thershold f_t`进度条中选择特征数据和目标的相关性阈值，低于阈值的特征将被丢弃。`Processed Data`中可点击`download `下载处理之后的数据。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231193226-0defd2b0-fe45-4dcb-8020-6ff7e32f9b37.jpg?raw=true" , width="400px" />
</p>

## 特征变量与特征变量相关性

---

在**Feature Engineering**模块- `Correlation of Features vs Features `模块下:

点击`Drop Collinear Features`按钮， 上传`.csv`文件之后，在`Drop Collinear Features`功能下`choose target`选择目标变量，显示特征与所选择目标的相关性系数热力图。`correlation method`中选择相关性方法中选择`pearson,spearman,kendall,` 皮尔森相关性系数、斯皮尔曼相关性系数、肯德尔相关性系数（类别变量）。在`correlation threshold`进度条中选择特征数据和特征数据之间的相关性阈值，高于阈值的两个特征将被筛选出来，丢弃其中与目标相关性更低的特征。在`Processed Data`中可点击`download `下载处理之后的数据。`is mask`功能选择是否将热力图进行掩码展示。`drop features`中显示丢弃的特征。`Processed Data`中可点击`download `下载处理之后的数据。

## 类别特征变量one-hot编码

---

在**Feature Engineering**模块- `One-hot Encoding Features `模块下:
点击`One-hot Encoding `按钮，上传`.csv`文件之后，在`One-hot encoding Features`中将会显示one-hot编码之后的数据，如类别特征`Sex`中值`female`和`male`将被转换为`0,1`和`1,0`，并删除旧特征`Sex`，创建新特征`Sex_female`和`Sex_male`添加到数据集中。`Processed Data`中可点击`download `下载处理之后的数据。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231193382-d0a374b6-420d-4735-b7a4-8468df3f8ea0.jpg?raw=true" , width="400px" />
</p>

## 特征变量重要性排序

---

在**Feature Engineering**模块- `Features Importance `模块下:
点击`Feature Importance`按钮，上传`.csv`文件之后，在`Choose Target`功能下选择目标特征。在`Selector`功能下选择`model`，其中`RandomForestClassifier`负责分类目标数据的特征重要性排序。`LassoRegressor, LinearRegressor,RandomForestRegressor, RidgeRegressor`负责连续目标数据的特征重要性排序。`Hyper Parameters`中选择不同算法的超参数，`cumulative importance `选择按照特征重要性从大到小排列加和的阈值，舍弃阈值之后的特征。点击`Embedded method`将使用嵌入法按照特征从到小的顺序依次添加训练模型，可视化不同重要性的特征对模型的影响，`cv`可选择交叉验证的折数。
点击`train`按钮，根据所选择的算法和超参数进行特征重要性排序，给出特征重要性计算表格，并绘制特征重要性直方图。`Processed Data`中可下载经过`dropped zero importance `的数据和经过`dropped low importance`的数据。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231182015-4e845d4a-7f2f-44e7-92a0-af0b4d9ab085.jpg?raw=true" , width="400px" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231182027-aa332363-0e36-42d6-80be-a672d7d5628f.jpg?raw=true" , width="400px" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231182056-02ac07fe-9c9f-4e11-b868-01d24a19cea8.jpg?raw=true" , width="400px" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231182067-18178c0f-bab0-4463-a4d7-a66b5244e3e4.jpg?raw=true" , width="400px" />
</p>

