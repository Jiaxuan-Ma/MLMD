<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231174459-96d33cdf-9f6f-4296-ba9f-31d11056ef12.jpg?raw=true" width="300px"  alt="MLMD"/>
</div>
</p>

The AI platform, MLMD (Machine Learning for Material Design) aims at utilizing general and frontier machine learning algrithm to accelerate the end-to-end material design with programming-free.

![1](https://github.com/Jiaxuan-Ma/MLMD/assets/61132191/98992016-e211-442a-aaec-2efc9ac8dc0f)


# Data Layoout


<img src="https://github.com/Jiaxuan-Ma/MLMD/assets/61132191/fa138ee2-b1a6-494d-a5de-741d1a54af14" width="800px">


# Remote access
The remote access is unstable, and we recommend deploying mlmd on a local laptop.
```
https://mgi-mlmd.streamlit.app/
```
Stable version
```
http://123.60.55.8/
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
