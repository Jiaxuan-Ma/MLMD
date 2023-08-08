# 类别目标变量分类预测

---

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231190770-6830545a-4c68-471d-92eb-32f0bb531178.jpg?raw=true" , width="400px" />
</p>


在`Prediction`模块下，单击`Classification`按钮。
  
进入**Classification**模块，页面弹出如下图所示的`.csv`文件上传框。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231178930-06bb0b95-1765-46bc-8011-d4932c7d7ea1.jpg?raw=true" , width="400px" />
</p>

---

**Classification**- `Data Information `模块:

上传数据之后，`Data Table`功能显示加载所上传的`.csv`文件的数据，可通过调节`rows`调整显示的数据表的行数。

``Features vs Targets``功能显示数据集的特征变量和目标变量，默认`.csv`文件中的最后一列为目标变量，可通过`input target`调节目标变量的个数。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231191093-dbb2a10a-a729-4239-9c9b-bc12872ba57c.jpg?raw=true" , width="400px" />
</p>

---

`Choose Target`功能选择目标特征

在`Regressor`功能下选择`model`，在`Hyper Parameters`中可调节每个算法的超参数

> 1:`DecisionTreeClassifier`- 决策树
> 
> 2:`LogisticRegression` - 逻辑回归
> 
> 3:`RandomForestClassifier` - 随机森林
> 
> 4:`SupportVectorClassifier` - 支持向量机

***集成学习***

> 1:`AdaBoostClassifier`- 自适应提升集成分类
>
> 2:`BaggingClassifier` - 自主聚合集成分类
>
> 3:`CatBoostClassifier` - 类别提升集成分类
>
> 4:`GradientBoosingClassifier` - 梯度提升集成分类
>
> 5:`LGBMClassifier` - 轻量梯度提升集成分类
>
> 6:`XGBClassifier` - 极限梯度提升集成分类

`data preprocess`中可选择
- `StandardScaler` - 标准化 
-  `MinMaxScaler` - 归一化 

`StandardScaler` - 标准化 

均值为0，标准差为1的标准化数学表达式为：

$$x^*=\frac{x-\mu}{\sigma}$$

其中$\mu=\frac{1}{n}\Sigma x_i$，注意此处标准差使用的是总体标准差$\sigma=\sqrt{\frac{\Sigma(x_i-\mu)^2}{n}}$


 `MinMaxScaler` - 归一化 

线性归一化数学表达式为：

$$x^*=\frac{x-min(x)}{max(x)-min(x)}$$

这种归一化方法比较适用在数值比较集中的情况。这种方法有个缺陷，如果$max$和$min$不稳定，很容易使得归一化结果不稳定，此时最好选用标准差标准化。


在`operator`可选择`train test split, cross val score`两种验证模型性能方法。 

- `train test split`：

点击`train`按钮，根据所选择的算法和超参数对**划分的测试集**类别预测，绘制混淆矩阵，并给真值和预测值表格，点击`download`可下载。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231191234-44c4c99c-6278-49c5-bde4-8e328c704961.jpg?raw=true" , width="400px" />
</p>


- `cross val score`：

可选择交叉验证的折数，推荐5~10折，点击`train`按钮，根据所选择的算法和超参数进行类别预测，给出预测结果的R2和真值预测值表格，点击`download`可下载。

