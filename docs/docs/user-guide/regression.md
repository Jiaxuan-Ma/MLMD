# 连续目标变量回归预测
---

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231191499-c3cf2b6c-ebaf-43b3-9b50-c1f85ae327b4.jpg?raw=true" , width="400px" />
</p>


在`Prediction`模块下，单击`Regression`按钮。
  
进入**Regression**模块，页面弹出如下图所示的`.csv`文件上传框。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231178930-06bb0b95-1765-46bc-8011-d4932c7d7ea1.jpg?raw=true" , width="400px" />
</p>

---

**Regression** - `Data Information `模块:

上传数据之后，`Data Table`功能显示加载所上传的`.csv`文件的数据，可通过调节`rows`调整显示的数据表的行数。

``Features vs Targets``功能显示数据集的特征变量和目标变量，默认`.csv`文件中的最后一列为目标变量，可通过`input target`调节目标变量的个数。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231191657-64592343-55d6-488c-a14b-04e6c5b36009.jpg?raw=true" , width="400px" />
</p>

---

`Choose Target`功能选择目标特征


在`Regressor`功能下选择`model`，在`Hyper Parameters`中可调节每个算法的超参数

> 1:`DecisionTreeRegressosr` - 决策树
> 
> 2:`KNeighborsRegressor` - K紧邻
> 
> 3:`LassoRegressor` - Lasso
> 
> 4:`LinearRegressor` - 线性回归
> 
> 5:`MLPRegressor` - 多层感知机
>
> 6:`RandomForestRegressor` - 随机森林
>
> 7:`RidgeRegressor` - 岭回归
>
> 8:`SupportRegressor` - 支持向量机

***集成学习***

> 1:`AdaboostRegressosr` - 自适应提升集成回归
>
> 2:`BaggingRegressor` - 自主聚合集成回归
>
> 3:`CatBoostRegressor` - 类别提升集成回归
>
> 4:`GradientBoostingRegressor` - 梯度提升集成回归
>
> 5:`LGBMRegressor` - 轻量梯度提升集成回归
>
> 6:`XGBRegressor` - 极限梯度提升集成回归


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



`operator`中可选择`train test split, cross val score, leave one out`三种验证模型性能方法。 

- `train test split`：

点击`train`按钮，根据所选择的算法和超参数对**划分的测试集**回归预测，绘制真值和预测值比较图，并给真值和预测值表格，点击`download`可下载。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231191757-2b2c19a9-d61a-4d19-aa9d-874fa1fa097d.jpg?raw=true" , width="400px" />
</p>



- `cross val score`：

可选择交叉验证的折数，推荐5~10折，点击`train`按钮，根据所选择的算法和超参数进行回归预测，给出预测结果的R2和真值预测值表格，点击`download`可下载。

- `leave one out`：

点击`train`按钮，根据所选择的算法和超参数进行回归预测，绘制**整个数据集**真值和预测值比较图，并给真值和预测值表格，点击`download`可下载。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231191832-c736f268-167a-4b7e-af80-28ebfa2bfc01.jpg?raw=true" , width="400px" />
</p>

