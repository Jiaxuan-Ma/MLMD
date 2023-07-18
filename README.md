<p align="center">
  <img src="https://user-images.githubusercontent.com/61132191/231174459-96d33cdf-9f6f-4296-ba9f-31d11056ef12.jpg?raw=true" width="300px"  alt="MLMD"/>
</div>
</p>

The **MLMD** platform (**M**achine **L**earning for **M**aterial **D**esign) for Material or Engineering aims at general and frontier machine learning algorithm with visualization. It is built on the traditional machine learning framework mostly based [scikit-learn](https://scikit-learn.org/stable/index.html), which provides the machine learning in python. 

| [Feature Engineering](https://mlmd.netlify.app/user-guide/feature%20engineering/) | [Regression](https://mlmd.netlify.app/user-guide/regression/) | [Classification](https://mlmd.netlify.app/user-guide/classification/) | [Active Learning](https://mlmd.netlify.app/user-guide/active%20learning/) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![law](https://user-images.githubusercontent.com/61132191/231174763-00e43b00-dac7-476d-ba7a-701241ea2337.png?raw=true)                              | ![law](https://user-images.githubusercontent.com/61132191/231175195-d65a2907-58d5-4488-bf27-4f78e89f1d4f.jpg?raw=true)                       | ![law](https://user-images.githubusercontent.com/61132191/231175281-0416b03d-8d6b-4d2a-abed-b21034a5bea2.jpg?raw=true)                              | ![law](https://user-images.githubusercontent.com/86995074/230322616-08fc629c-1858-42e7-8795-57fc8d076339.png?raw=true)    


Check out [help document](https://mlmd.netlify.app/) for more information
# Remote access

```
https://ml4material.streamlit.app/
```

# Local Deploy

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
