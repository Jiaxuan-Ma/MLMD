<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231174459-96d33cdf-9f6f-4296-ba9f-31d11056ef12.jpg?raw=true" width="300px"  alt="MLMD"/>
</div>
</p>

The **MLMD** platform (**M**achine **L**earning for **M**aterial **D**esign) for Material or Engineering aims at general and frontier machine learning algorithm with visualization. It is built on the traditional machine learning framework mostly based [scikit-learn](https://scikit-learn.org/stable/index.html), which provides the machine learning in python. 

# MLMD框架
![图片1](https://github.com/Jiaxuan-Ma/MLMD/assets/61132191/fa3fd53d-e922-4d7e-b4a3-ed7dc5ab62d6)

# 功能
## 单目标/多目标 嵌入代理模型的材料成分（工艺）设计流程

*训练代理模型需要大量的初始样本点，一般仅需要一次循环迭代*
![图片1](https://github.com/Jiaxuan-Ma/MLMD/assets/61132191/dac35e71-1576-4142-ba68-fb1cf150b801)

## 单目标/多目标 基于贝叶斯的材料成分（工艺）设计流程

*训练代理模型需要少量的初始样本点，一般需要多次循环迭代*
![图片2](https://github.com/Jiaxuan-Ma/MLMD/assets/61132191/64762434-abb6-41ac-9ab5-0394a5e32f88)

## 单目标/多目标 嵌入代理模型的迁移学习的材料成分（工艺）设计流程

**上线日期待定**
![图片3](https://github.com/Jiaxuan-Ma/MLMD/assets/61132191/a723f604-f7fa-4c69-9bab-5e7892c990f3)


Check out [help document](https://mlmd.netlify.app/) for more information
# Remote access

```
https://ml4material.streamlit.app/
```

# Local deploy

1. Install [anaconda](https://www.anaconda.com/). 
2. Create virtual envrionment
```
conda create -n MLMD python=3.9
```
1. Install dependent libraries
```
pip install -r requirements.txt
```
1. Run MLMD 
```
streamlit run MLMD.py
```

# License
MLMD platform is released under MIT License. See [License](https://github.com/Jiaxuan-Ma/Machine-Learning-for-Material-Design/blob/main/LICENSE) for details
