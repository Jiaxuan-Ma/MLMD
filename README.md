<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231174459-96d33cdf-9f6f-4296-ba9f-31d11056ef12.jpg?raw=true" width="300px"  alt="MLMD"/>
</div>
</p>

# MLMD: a programming-free AI platform to predict and design materials

MLMD is capable of effectively discovering novel materials with high-potential advanced properties end-to-end, utilizing model inference, surrogate optimization, 
and even working in situations of data scarcity based on active learning. Additionally, it integrates data analysis, descriptor refactoring, hyper-parameters auto-optimizing, and
properties prediction. It also provides a web-based friendly interface without need programming and can be used anywhere, anytime.

材料基因工程的研发理念深刻变革了材料研发范式，提高了新材料的研发效率，降低了研发成本。材料基因工程研发理念的核心是材料信息学，人工智能技术是材料信息学的核心工具。但是材料领域试验科研人员往往缺乏编程基础，这一门槛限制了利用材料基因理念辅助材料设计的发展。因此，我们开发了一个基于web端的界面友好的AI材料设计平台MLMD(Machine Learning for Materials Design)。平台集成了材料信息学常用的机器学习算法，包含异常值筛选、特征描述符重构、特征相关性分析和特征重要性排序等常用的特征工程算法，并可实现回归预测、分类预测和聚类等。平台还针对材料设计的需要，集成了随机优化算法用于单目标或者多目标特性的材料设计，开发了[贝叶斯主动学习模块](https://colab.research.google.com/drive/1OSc-phxm7QLOm8ceGJiIMGGz9riuwP6Q?usp=sharing)和基于迁移学习材料设计模块解决材料领域小数据的问题。

## Overview and architecture of MLMD（MLMD的架构和功能模块）

<img src="https://github.com/Jiaxuan-Ma/MLMD/assets/61132191/ec6835e7-7dcb-4c82-b4e2-c37c8ff9251f" width="500px">

## Flowcharts of materials design in MLMD platform（单目标/多目标材料设计流程图）

<img src="https://github.com/Jiaxuan-Ma/MLMD/assets/61132191/a5785637-fec5-4d20-9b58-3c9437f2aadb" width="500px">

## Uploaded data layoout（上传的数据形式）

<img src="https://github.com/Jiaxuan-Ma/MLMD/assets/61132191/fa138ee2-b1a6-494d-a5de-741d1a54af14" width="500px">

---


## Local deploy（本地部署）

1. Install [anaconda](https://www.anaconda.com/) on local machine(本地电脑安装anaconda)
  
2. Create virtual envrionment(使用conda创建mlmd环境)
```
conda create -n mlmd python=3.10
```
3. Git clone the MLMD code library (or downloaded manually)(git clone或者手动下载MLMD库)

4. Install required libraries in the MLMD directory(在MLMD目录安装依赖库)
```
pip install -r requirements.txt
```
5. Run MLMD(运行MLMD)
```
streamlit run MLMD.py
```

## Remote access（远程访问）

**Given the limitation of computational resources, we strongly recommend deploying MLMD on your local machine!!!**

Stable version
```
matdesign.top
```
## Contact
WeChat Official Accounts： 

<img width="100" alt="截屏2024-03-30 00 22 30" src="https://github.com/Jiaxuan-Ma/MLMD/assets/61132191/6f9f99e9-ee24-426a-8198-76f5a1ac7460">
AI for Mechanics

<img width="100" alt="截屏2024-03-30 00 24 00" src="https://github.com/Jiaxuan-Ma/MLMD/assets/61132191/c518c16f-fcd5-45a8-9a65-6cad1bb72d53">
SciMindBin

## License
This code is licensed under the MIT license. Feel free to use all or portions for your research or related projects so long as you provide the following citation information:

Ma, J.∔, Cao, B.∔, Dong, S. et al. MLMD: a programming-free AI platform to predict and design materials. npj Comput Mater 10, 59 (2024). https://doi.org/10.1038/s41524-024-01243-4
```
@article{Ma2024a,
  title = {MLMD: A Programming-Free AI Platform to Predict and Design Materials},
  shorttitle = {MLMD},
  author = {Ma, Jiaxuan and Cao, Bin and Dong, Shuya and Tian, Yuan and Wang, Menghuan and Xiong, Jie and Sun, Sheng},
  year = {2024},
  month = mar,
  journal = {npj Computational Materials},
  volume = {10},
  number = {1},
  pages = {59},
  issn = {2057-3960},
  doi = {10.1038/s41524-024-01243-4}
}
```
