# 类别目标变量分类预测

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/prediction/Mjx-20230409-5.jpg?raw=true" , width="400px" />
</p>


在`Prediction`模块下，单击`Regression`按钮。
  
进入`Regression`模块，页面弹出如下图所示的`.csv`文件上传框。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/feature%20engineering/Mjx-20230408-5.jpg?raw=true" , width="400px" />
</p>


`Regression`- `Data Information `模块:

上传数据之后，`Data Table`功能显示加载所上传的`.csv`文件的数据，可通过调节`rows`调整显示的数据表的行数。

``Features vs Targets``功能显示数据集的特征变量和目标变量，默认`.csv`文件中的最后一列为目标变量，可通过`input target`调节目标变量的个数。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/prediction/Mjx-20230409-6.jpg?raw=true" , width="400px" />
</p>



`Choose Target`功能选择目标特征

在`Regressor`功能下选择`model`，在`Hyper Parameters`中可调节每个算法的超参数

> `DecisionTreeClassifier`
> 
> `LogisticRegression`
> 
> `RandomForestClassifier`
> 
> `SupportVectorClassifier`


在`operator`可选择`train test split, cross val score`两种验证模型性能方法。 

`train test split`：

点击`train`按钮，根据所选择的算法和超参数对**划分的测试集**类别预测，绘制混淆矩阵，并给真值和预测值表格，点击`download`可下载。

<p align="center">
  <img src="https://github.com/Jiaxuan-Ma/MLMDMarket/blob/main/prediction/Mjx-20230409-7.jpg?raw=true" , width="400px" />
</p>



`cross val score`：

可选择交叉验证的折数，推荐5~10折，点击`train`按钮，根据所选择的算法和超参数进行类别预测，给出预测结果的R2和真值预测值表格，点击`download`可下载。
