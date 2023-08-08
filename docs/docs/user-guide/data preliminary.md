# 数据表格智能分析和可视化

---
该功能模块为特征工程提供初步的数据智能分析和可视化展示，主要实现特征变量和目标变量在数据集中的分布和关系可视化。

## 数据表格智能分析

---
用户登录时，在`Data Preliminar`功能模块下，单击`Data Profiling`按钮。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231178901-1b5e3526-30ba-4366-81ef-8f74781330f8.jpg?raw=true" , width="400px" />
</p>

进入`Data Profiling`模块，页面弹出如下图所示的`.csv`文件上传框。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231178930-06bb0b95-1765-46bc-8011-d4932c7d7ea1.jpg?raw=true" , width="400px" />
</p>

上传数据之后，页面显示数据的智能分析报告。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231179512-8cbf9dbd-576b-47a5-9ec3-e123194b0756.jpg?raw=true" , width="400px" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231179522-0a5d002a-0ee7-445d-a940-5cbcce1f5ca3.jpg?raw=true" , width="400px" />
</p>



<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231179563-b0cdd400-1ce0-4c9b-873f-cd5b21cae346.jpg?raw=true" , width="400px" />
</p>

## 数据变量关系可视化分析
---

用户登录时，在`Data Preliminar`功能模块下，单击`Data Visualization`按钮。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231180122-48b30a78-ba62-460c-b3e3-8a5ce082cd5b.jpg?raw=true" , width="400px" />
</p>

### 数据表格信息
---

进入`Data Profiling`模块，上传数据之后，`Data Table`功能显示加载所上传的`.csv`文件的数据，可通过调节`rows`调整显示的数据表的行数。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231180325-abd85f39-5495-4f6c-a5df-20292f5d3922.jpg?raw=true" , width="400px" />
</p>

### 数据统计信息
---

`Data Statistics`功能显示所上传数据的统计信息，点击`download`可进行下载

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231180342-a1406efd-0899-4c1c-ba0e-4fa6d9383bb8.jpg?raw=true" , width="400px" />
</p>

### 选择目标变量
---

`Features vs Targets`功能显示数据集的特征变量和目标变量，默认`.csv`文件中的最后一列为目标变量，可通过`input target`调节目标变量的个数。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231180375-bd881cc3-87cb-47b0-b667-d5d4110758e8.jpg?raw=true" , width="400px" />
</p>

### 特征变量分布

---

`Feature Statistics Distribution`功能显示每个特征变量分布统计直方图并给出核密度估计曲线，可通过`Plot parameters`功能调节图像的颜色、字体、标题和刻度大小。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231180657-3a1c8288-a619-432a-aa3b-21a1efbeed0e.jpg?raw=true" , width="400px" />
</p>

### 目标变量分布
---

`Target Statistics Distribution`功能显示每个特征变量分布统计直方图并给出核密度估计曲线，可通过`Plot parameters`功能调节图像的颜色、字体、标题和刻度大小。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231180682-0cb76fe7-5a43-41b6-b4e6-2f8a46501c13.jpg?raw=true" , width="400px" />
</p>

### 特征变量配方分布
---

`Feature Recipe Distribution`功能按照数据集中特征的顺序统计每个特征在样本中的数量，从而得知目标的常规配方，可通过`Plot parameters`功能调节图像的颜色、字体、标题和刻度大小。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231180701-3dd3b1b5-ceab-483f-ae17-6cd0aaac7d3e.jpg?raw=true" , width="400px" />
</p>

### 特征变量数据集分布
---

`Distribution of Feature in Dataset`功能统计特征变量在数据集中的分布情况，可通过`Plot parameters`功能调节图像的颜色、字体、标题和刻度大小。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231180730-fe3c5000-db26-49b4-a265-836a9b516c83.jpg?raw=true" , width="400px" />
</p>

### 特征变量与目标变量

---

`Features and Targets`功能显示特征变量和目标变量的关系，可通过`Plot parameters`功能调节图像的颜色、字体、标题和刻度大小。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231180750-abe6389d-faf5-48b5-b2a3-e209a6a2a5b7.jpg?raw=true" , width="400px" />
</p>

### 目标变量与目标变量

---

`Tagrets and Targets`功能显示特征变量和目标变量的关系，可通过`Plot parameters`功能调节图像的颜色、字体、标题和刻度大小 。

如果是多目标数据，`Tagrets and Targets`功能显示目标变量和目标变量的关系，可通过`Plot parameters`功能调节图像的颜色、字体、标题和刻度大小 。

<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231180773-438ef9ea-da77-40e3-ba23-37988d2d8f35.jpg?raw=true" , width="400px" />
</p>


