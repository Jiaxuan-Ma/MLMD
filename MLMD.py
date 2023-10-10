'''
Runs the streamlit app
Call this file in the terminal via `streamlit run app.py`
'''
from multiprocessing import freeze_support
import threading


import shap

import streamlit as st

from streamlit_extras.colored_header import colored_header
from streamlit_option_menu import option_menu
from streamlit_extras.badges import badge
from streamlit_shap import st_shap
from streamlit_card import card

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split as TTS
from sklearn.model_selection import cross_val_score as CVS
from sklearn.model_selection import cross_validate as CV

from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn import tree

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression as MIR
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.manifold import TSNE
from PIL import Image


import xgboost as xgb
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
import Bgolearn.BGOsampling as BGOS

from typing import Optional
import graphviz

import shap
import matplotlib.pyplot as plt
import pickle
from utils import *
from streamlit_extras.badges import badge
from sklearn.gaussian_process.kernels import RBF
import warnings

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.algorithms.moo.nsga2 import SBX as nsgaSBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination

from sko.PSO import PSO
from sko.DE import DE
# from sko.AFSA import AFSA
from sko.SA import SAFast

# import sys
from prettytable import PrettyTable


from algorithm.TrAdaboostR2 import TrAdaboostR2
from algorithm.mobo import Mobo4mat
import scienceplots

st.set_page_config(
        page_title="MLMD",
        page_icon="🍁",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items={
        })

sysmenu = '''
<style>
MainMenu {visibility:hidden;}
footer {visibility:hidden;}
'''

# https://icons.bootcss.com/
st.markdown(sysmenu,unsafe_allow_html=True)

with st.sidebar:
    st.write('''
    技术支持：马家轩|Jiaxuan Ma
             
    联系方式：jiaxuanma.shu@gmail.com
    ''')
    select_option = option_menu("MLMD", ["平台主页", "基础功能", "特征工程","聚类降维", "回归预测", "分类预测", "主动学习","迁移学习", "代理优化", "其他"],
                    icons=['house', 'clipboard-data', 'menu-button-wide','circle','bezier2', 'subtract', 'arrow-repeat', 'app', 'microsoft'],
                    menu_icon="boxes", default_index=0)
if select_option == "平台主页":
    st.write('''![](https://user-images.githubusercontent.com/61132191/231174459-96d33cdf-9f6f-4296-ba9f-31d11056ef12.jpg?raw=true)''')


    colored_header(label="材料设计的机器学习平台",description="Machine Learning for Material Design",color_name="violet-90")

    st.markdown(
    '''
    The **MLMD** platform (**M**achine **L**earning for **M**aterial **D**esign) for Material or Engineering aims at utilizing general and frontier machine learning algrithm to accelerate the material design with no-programming. \n
    材料基因组工程理念的发展将会大幅度提高新材料的研发效率、缩短研发周期、降低研发成本、全面加速材料从设计到工程化应用的进程。
    因此**MLMD**旨在为材料试验科研人员提供快速上手，无编程的机器学习算法平台，致力于材料试验到材料设计的一体化。
    ''')
    colored_header(label="数据布局",description="only support `.csv` file",color_name="violet-90")

    st.write('''![](https://github.com/Jiaxuan-Ma/Jiaxuan-Ma/assets/61132191/470e2fc4-0e99-4a28-afc3-1c93c44758da?raw=true)''')
    st.write(
        '''*为了保证合金的各个元素的质量分数总和为100%, 因此使用**代理优化**模块时需要去掉基元素列*''')
    colored_header(label="致谢",description="",color_name="violet-90")

    st.markdown(
    '''
    #### 贡献者
    **课题组**: [上海大学 材料信息学与力学信息学实验室(MMIL)](http://www.sshome.space/MMIL/PI/)

    **主要开发人**：马家轩（博士在读）

    **指导教师**：孙升（研究员） 

    **参与人**：
    
    熊杰 （讲师）
    田原 （博士）

    #### 资助
    国家科技部重点研发计划(No. 2022YFB3707803)
    
    ''')

elif select_option == "基础功能":
    with st.sidebar:
        sub_option = option_menu(None, ["数据库建设", "数据可视化"])
    if sub_option == "数据库建设":
        colored_header(label="数据库建设",description=" ",color_name="violet-90")
        col1, col2 = st.columns([2,2])
        with col1:
            df = pd.read_csv('./data/in.csv')
            st.write("高熵合金数据库")
            st.write(df.head())
            tmp_download_link = download_button(df , f'预测结果.csv', button_text='download')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
        with col2:
            df = pd.read_csv('./data/in.csv')
            st.write("钢数据库")
            st.write(df.head())
            tmp_download_link = download_button(df , f'预测结果.csv', button_text='download')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

        col1, col2 = st.columns([2,2])
        with col1:
            df = pd.read_csv('./data/in.csv')
            st.write("高熵合金数据库")
            image = Image.open('./data/fig1.png')
            st.image(image, width=280,caption='test size')
            tmp_download_link = download_button(df , f'预测结果.csv', button_text='download')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
        with col2:
            df = pd.read_csv('./data/in.csv')
            st.write("钢数据库")
            st.write(df.head())
            tmp_download_link = download_button(df , f'预测结果.csv', button_text='download')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
            # tmp_download_link = download_button(result_data, f'预测结果.csv', button_text='download')
            # st.markdown(tmp_download_link, unsafe_allow_html=True)


    elif sub_option == "数据可视化":

        colored_header(label="数据可视化",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv` file", type=['csv'], label_visibility="collapsed")
        if file is None:
            table = PrettyTable(['上传文件名称', '名称','数据说明'])
            table.add_row(['file_1','dataset','数据集'])
            st.write(table)
        if file is not None:
            df = pd.read_csv(file)
            # check NaN
            check_string_NaN(df)
            
            colored_header(label="数据信息",description=" ",color_name="violet-70")

            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="数据初步统计",description=" ",color_name="violet-30")

            st.write(df.describe())

            tmp_download_link = download_button(df.describe(), f'数据统计.csv', button_text='download')
            
            st.markdown(tmp_download_link, unsafe_allow_html=True)

            colored_header(label="特征变量和目标变量", description=" ",color_name="violet-70")
            
            target_num = st.number_input('目标变量数量', min_value=1, max_value=10, value=1)
            col_feature, col_target = st.columns(2)
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())

            # colored_header(label="特征变量统计分布", description=" ",color_name="violet-30")

            # feature_selected_name = st.selectbox('选择特征变量',list(features))
        
            # feature_selected_value = features[feature_selected_name]
            # plot = customPlot()
            # col1, col2 = st.columns([1,3])
            # with col1:  
            #     with st.expander("绘图参数"):
            #         options_selected = [plot.set_title_fontsize(1),plot.set_label_fontsize(2),
            #                     plot.set_tick_fontsize(3),plot.set_legend_fontsize(4),plot.set_color('line color',6,5),plot.set_color('bin color',0,6)]
            # with col2:
            #     plot.feature_hist_kde(options_selected,feature_selected_name,feature_selected_value)
            colored_header(label="特征变量在数据集中的分布", description=" ",color_name="violet-30")
            feature_selected_name = st.selectbox('选择特征变量', list(features),1)
            feature_selected_value = features[feature_selected_name]
            plot = customPlot()
            col1, col2 = st.columns([1,3])
            with col1:  
                with st.expander("绘图参数"):
                    options_selected = [plot.set_title_fontsize(18),plot.set_label_fontsize(19),
                                plot.set_tick_fontsize(20),plot.set_legend_fontsize(21), plot.set_color('bin color', 0, 22)]
            with col2:
                plot.feature_distribution(options_selected,feature_selected_name,feature_selected_value)
            
            with col1:  
                with st.expander("绘图参数"):
                    options_selected = [plot.set_title_fontsize(1),plot.set_label_fontsize(2),
                                plot.set_tick_fontsize(3),plot.set_legend_fontsize(4),plot.set_color('line color',6,5),plot.set_color('bin color',0,6)]
            with col2:
                plot.feature_hist_kde(options_selected,feature_selected_name,feature_selected_value)

            #=========== Targets visulization ==================

            colored_header(label="目标变量统计分布", description=" ",color_name="violet-30")

            target_selected_name = st.selectbox('选择目标变量',list(targets))

            target_selected_value = targets[target_selected_name]
            plot = customPlot()
            col1, col2 = st.columns([1,3])
            with col1:  
                with st.expander("绘图参数"):
                    options_selected = [plot.set_title_fontsize(7),plot.set_label_fontsize(8),
                                plot.set_tick_fontsize(9),plot.set_legend_fontsize(10), plot.set_color('line color',6,11), plot.set_color('bin color',0,12)]
            with col2:
                plot.target_hist_kde(options_selected,target_selected_name,target_selected_value)

            #=========== Features analysis ==================

            colored_header(label="特征变量配方（合金成分）", description=" ",color_name="violet-30")

            feature_range_selected_name = st.slider('选择特征变量个数',1,len(features.columns), (1,2))
            min_feature_selected = feature_range_selected_name[0]-1
            max_feature_selected = feature_range_selected_name[1]
            feature_range_selected_value = features.iloc[:,min_feature_selected: max_feature_selected]
            data_by_feature_type = df.groupby(list(feature_range_selected_value))
            feature_type_data = create_data_with_group_and_counts(data_by_feature_type)
            IDs = [str(id_) for id_ in feature_type_data['ID']]
            Counts = feature_type_data['Count']
            col1, col2 = st.columns([1,3])
            with col1:  
                with st.expander("绘图参数"):
                    options_selected = [plot.set_title_fontsize(13),plot.set_label_fontsize(14),
                                plot.set_tick_fontsize(15),plot.set_legend_fontsize(16),plot.set_color('bin color',0, 17)]
            with col2:
                plot.featureSets_statistics_hist(options_selected,IDs, Counts)



            # colored_header(label="特征变量和目标变量关系", description=" ",color_name="violet-30")
            # col1, col2 = st.columns([1,3])
            # with col1:  
            #     with st.expander("绘图参数"):
            #         options_selected = [plot.set_title_fontsize(23),plot.set_label_fontsize(24),
            #                     plot.set_tick_fontsize(25),plot.set_legend_fontsize(26),plot.set_color('scatter color',0, 27),plot.set_color('line color',6,28)]
            # with col2:
            #     plot.features_and_targets(options_selected,df, list(features), list(targets))
            
            # # st.write("### Targets and Targets ")
            # if targets.shape[1] != 1:
            #     colored_header(label="目标变量和目标变量关系", description=" ",color_name="violet-30")
            #     col1, col2 = st.columns([1,3])
            #     with col1:  
            #         with st.expander("绘图参数"):
            #             options_selected = [plot.set_title_fontsize(29),plot.set_label_fontsize(30),
            #                         plot.set_tick_fontsize(31),plot.set_legend_fontsize(32),plot.set_color('scatter color',0, 33),plot.set_color('line color',6,34)]
            #     with col2:
            #         plot.targets_and_targets(options_selected,df, list(targets))
        st.write('---')

elif select_option == "特征工程":
    with st.sidebar:
        sub_option = option_menu(None, ["空值处理","特征转换", "特征唯一值处理", "特征和特征相关性", "特征和目标相关性", "One-hot编码", "特征重要性"])

    if sub_option == "空值处理":
        colored_header(label="空值处理",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
        if file is None:
            table = PrettyTable(['上传文件名称', '名称','数据说明'])
            table.add_row(['file_1','dataset','数据集'])
            st.write(table)
        if file is not None:
        # colored_header(label="数据信息",description=" ",color_name="violet-70")
        # with st.expander('数据信息'):
            df = pd.read_csv(file)
            # check NuLL 
            null_columns = df.columns[df.isnull().any()]
            if len(null_columns) == 0:
                st.error('No missing features!')
                st.stop()
                
            colored_header(label="数据信息", description=" ",color_name="violet-70")
            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="目标变量和特征变量",description=" ",color_name="violet-70")

            target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)
            
            col_feature, col_target = st.columns(2)
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())
        
            # sub_sub_option = option_menu(None, ["丢弃空值", "填充空值"])
            colored_header(label="选择方法",description=" ",color_name="violet-70")
            sub_sub_option = option_menu(None, ["丢弃空值", "填充空值"],
                    icons=['house',  "list-task"],
                    menu_icon="cast", default_index=0, orientation="horizontal")
            if sub_sub_option == "丢弃空值":
                fs = FeatureSelector(features, targets)
                missing_threshold = st.slider("丢弃阈值(空值占比)",0.001, 1.0, 0.5)
                fs.identify_missing(missing_threshold)
                fs.features_dropped_missing = fs.features.drop(columns=fs.ops['missing'])
                
                data = pd.concat([fs.features_dropped_missing, targets], axis=1)
                st.write(data)
                tmp_download_link = download_button(data, f'空值丢弃数据.csv', button_text='download')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write('%d features with $\gt$ %0.2f missing threshold.\n' % (len(fs.ops['missing']), fs.missing_threshold))
                plot = customPlot()

                with st.expander('绘图'):
                    col1, col2 = st.columns([1,3])
                    with col1:
                        options_selected = [plot.set_title_fontsize(1),plot.set_label_fontsize(2),
                                    plot.set_tick_fontsize(3),plot.set_legend_fontsize(4),plot.set_color('bin color',19,5)]
                    with col2:
                        plot.feature_missing(options_selected, fs.record_missing, fs.missing_stats)
                st.write('---')
            if sub_sub_option == "填充空值":
                fs = FeatureSelector(features, targets)
                missing_feature_list = fs.features.columns[fs.features.isnull().any()].tolist()
                with st.container():
                    fill_method = st.selectbox('填充方法',('常值', '随机森林算法'))
                
                if fill_method == '常值':

                    missing_feature = st.multiselect('丢失值特征',missing_feature_list,missing_feature_list[-1])
                    
                    option_filled = st.selectbox('均值',('均值','常数','中位数','众数'))
                    if option_filled == '均值':
                        # fs.features[missing_feature] = fs.features[missing_feature].fillna(fs.features[missing_feature].mean())
                        imp = SimpleImputer(missing_values=np.nan,strategy= 'mean')

                        fs.features[missing_feature] = imp.fit_transform(fs.features[missing_feature])
                    elif option_filled == '常数':
                        # fs.features[missing_feature] = fs.features[missing_feature].fillna(0)
                        fill_value = st.number_input('输入数值')
                        imp = SimpleImputer(missing_values=np.nan, strategy= 'constant', fill_value = fill_value)
                        
                        fs.features[missing_feature] = imp.fit_transform(fs.features[missing_feature])
                    elif option_filled == '中位数':
                        # fs.features[missing_feature] = fs.features[missing_feature].fillna(0)
                        imp = SimpleImputer(missing_values=np.nan, strategy= 'median')
                        
                        fs.features[missing_feature] = imp.fit_transform(fs.features[missing_feature])
                    elif option_filled == '众数':

                        imp = SimpleImputer(missing_values=np.nan, strategy= 'most_frequent')
                        
                        fs.features[missing_feature] = imp.fit_transform(fs.features[missing_feature])  

                    data = pd.concat([fs.features, targets], axis=1)
                else:
                    with st.expander('超参数'):
                        num_estimators = st.number_input('number estimators',1, 10000, 100)
                        criterion = st.selectbox('criterion',('squared_error','absolute_error','friedman_mse','poisson'))
                        max_depth = st.number_input('max depth',1, 1000, 5)
                        min_samples_leaf = st.number_input('min samples leaf', 1, 1000, 5)
                        min_samples_split = st.number_input('min samples split', 1, 1000, 5)
                        random_state = st.checkbox('random state 1024',True)


                    option_filled = st.selectbox('均值',('均值','常数','中位数','众数'))
                    if option_filled == '均值':
                        feature_missing_reg = fs.features.copy()
                        null_columns = feature_missing_reg.columns[feature_missing_reg.isnull().any()] 
            
                        null_counts = feature_missing_reg.isnull().sum()[null_columns].sort_values() 
                        null_columns_ordered = null_counts.index.tolist() 
        
                        for i in null_columns_ordered:
        
                            df = feature_missing_reg
                            fillc = df[i]
            
                            df = pd.concat([df.iloc[:,df.columns != i], pd.DataFrame(targets)], axis=1)
                    
                            df_temp_fill = SimpleImputer(missing_values=np.nan,strategy= 'mean').fit_transform(df)

                            YTrain = fillc[fillc.notnull()]
                            YTest = fillc[fillc.isnull()]
                            XTrain = df_temp_fill[YTrain.index,:]
                            XTest = df_temp_fill[YTest.index,:]

                            rfc = RFR(n_estimators=num_estimators,criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,
                                    min_samples_split=min_samples_split,random_state=random_state)

                            rfc = rfc.fit(XTrain, YTrain)
                            YPredict = rfc.predict(XTest)
            
                            feature_missing_reg.loc[feature_missing_reg[i].isnull(), i] = YPredict

                    elif option_filled == '常数':

                        fill_value = st.number_input('输入数值')
                        feature_missing_reg = fs.features.copy()
                        
                        null_columns = feature_missing_reg.columns[feature_missing_reg.isnull().any()] 
            
                        null_counts = feature_missing_reg.isnull().sum()[null_columns].sort_values() 
                        null_columns_ordered = null_counts.index.tolist() 
        
                        for i in null_columns_ordered:

                            df = feature_missing_reg
                            fillc = df[i]
            
                            df = pd.concat([df.iloc[:,df.columns != i], pd.DataFrame(targets)], axis=1)

                            df_temp_fill = SimpleImputer(missing_values=np.nan, strategy= 'constant', fill_value = fill_value).fit_transform(df)

                            YTrain = fillc[fillc.notnull()]
                            YTest = fillc[fillc.isnull()]
                            XTrain = df_temp_fill[YTrain.index,:]
                            XTest = df_temp_fill[YTest.index,:]

                            rfc = RFR(n_estimators=num_estimators,criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,
                                    min_samples_split=min_samples_split,random_state=random_state)

                            rfc = rfc.fit(XTrain, YTrain)
                            YPredict = rfc.predict(XTest)

                            feature_missing_reg.loc[feature_missing_reg[i].isnull(), i] = YPredict
                        
                    elif option_filled == '中位数':
                        feature_missing_reg = fs.features.copy()
                    
                        null_columns = feature_missing_reg.columns[feature_missing_reg.isnull().any()] 
                
                        null_counts = feature_missing_reg.isnull().sum()[null_columns].sort_values() 
                        null_columns_ordered = null_counts.index.tolist() 

                        for i in null_columns_ordered:
        
                            df = feature_missing_reg
                            fillc = df[i]
                
                            df = pd.concat([df.iloc[:,df.columns != i], pd.DataFrame(targets)], axis=1)
        
                            df_temp_fill = SimpleImputer(missing_values=np.nan,strategy= 'median').fit_transform(df)

                            YTrain = fillc[fillc.notnull()]
                            YTest = fillc[fillc.isnull()]
                            XTrain = df_temp_fill[YTrain.index,:]
                            XTest = df_temp_fill[YTest.index,:]

                            rfc = RFR(n_estimators=num_estimators,criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,
                                    min_samples_split=min_samples_split,random_state=random_state)

                            rfc = rfc.fit(XTrain, YTrain)
                            YPredict = rfc.predict(XTest)
        
                            feature_missing_reg.loc[feature_missing_reg[i].isnull(), i] = YPredict
            
                    elif option_filled == '众数':

                        feature_missing_reg = fs.features.copy()
                    
                        null_columns = feature_missing_reg.columns[feature_missing_reg.isnull().any()] 
                
                        null_counts = feature_missing_reg.isnull().sum()[null_columns].sort_values() 
                        null_columns_ordered = null_counts.index.tolist() 

                        for i in null_columns_ordered:

                            df = feature_missing_reg
                            fillc = df[i]
                
                            df = pd.concat([df.iloc[:,df.columns != i], pd.DataFrame(targets)], axis=1)
            
                            df_temp_fill = SimpleImputer(missing_values=np.nan,strategy= 'most_frequent').fit_transform(df)

                            YTrain = fillc[fillc.notnull()]
                            YTest = fillc[fillc.isnull()]
                            XTrain = df_temp_fill[YTrain.index,:]
                            XTest = df_temp_fill[YTest.index,:]

                            rfc = RFR(n_estimators=num_estimators,criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,
                                    min_samples_split=min_samples_split,random_state=random_state)
                            rfc = rfc.fit(XTrain, YTrain)
                            YPredict = rfc.predict(XTest)
                            
                            feature_missing_reg.loc[feature_missing_reg[i].isnull(), i] = YPredict

                    data = pd.concat([feature_missing_reg, targets], axis=1)

                st.write(data)

                tmp_download_link = download_button(data, f'空值填充数据.csv', button_text='download')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write('---')
    
    elif sub_option == "特征唯一值处理":
        colored_header(label="特征唯一值处理",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
        if file is None:
            table = PrettyTable(['上传文件名称', '名称','数据说明'])
            table.add_row(['file_1','dataset','数据集'])
            st.write(table)
        if file is not None:
            colored_header(label="数据信息",description=" ",color_name="violet-70")

            df = pd.read_csv(file)
            # 检测缺失值
            check_string_NaN(df)

            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

            target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)
            
            col_feature, col_target = st.columns(2)
            
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())

            colored_header(label="丢弃唯一值特征变量",description=" ",color_name="violet-70")
            fs = FeatureSelector(features, targets)
            plot = customPlot() 

            col1, col2 = st.columns([1,3])
            with col1:
                
                fs.identify_nunique()
                option_counts = st.slider('丢弃唯一值特征数量',0, int(fs.unique_stats.max())-1,1)
                st.write(fs.unique_stats)
            with col2:

                fs.identify_nunique(option_counts)
                fs.features_dropped_single = fs.features.drop(columns=fs.ops['single_unique'])
                data = pd.concat([fs.features_dropped_single, targets], axis=1)
                st.write(fs.features_dropped_single)
                
                tmp_download_link = download_button(data, f'丢弃唯一值特征数据.csv', button_text='download')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write('%d features $\leq$  %d unique value.\n' % (len(fs.ops['single_unique']),option_counts))
    
            with st.expander('绘图'):
                col1, col2 = st.columns([1,3])
                with col1:
                    options_selected = [plot.set_title_fontsize(6),plot.set_label_fontsize(7),
                                plot.set_tick_fontsize(8),plot.set_legend_fontsize(9),plot.set_color('bin color',19,10)]
                with col2:
                    plot.feature_nunique(options_selected, fs.record_single_unique,fs.unique_stats)     
                
            st.write('---')

    elif sub_option == "特征转换":
        colored_header(label="特征变换",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
        if file is None:
            table = PrettyTable(['Composition'])
            table.add_row(['(Fe0.76B0.24)96Nb4'])
            st.write(table)
        if file is not None:
            colored_header(label="数据信息",description=" ",color_name="violet-70")

            df = pd.read_csv(file)
            df_nrow = df.head()
            st.write(df_nrow)
            option = st.selectbox('option',['Alloy', 'Inorganic'])
            button = st.button('Transform', use_container_width=True)
            if button:
                df = feature_transform(df, option)
                st.write(df.head())
                tmp_download_link = download_button(df, f'特征转换数据.csv', button_text='download')
                st.markdown(tmp_download_link, unsafe_allow_html=True)      


    elif sub_option == "特征和特征相关性":
        colored_header(label="特征和特征相关性",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
        if file is None:
            table = PrettyTable(['上传文件名称', '名称','数据说明'])
            table.add_row(['file_1','dataset','数据集'])
            st.write(table)
        if file is not None:
            df = pd.read_csv(file)
            # 检测缺失值
            check_string_NaN(df)
            colored_header(label="数据信息", description=" ",color_name="violet-70")
            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

            target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)
            
            col_feature, col_target = st.columns(2)
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())
        
            colored_header(label="丢弃双线性特征变量",description=" ",color_name="violet-30")
            fs = FeatureSelector(features, targets)
            plot = customPlot() 

            target_selected_option = st.selectbox('选择目标', list(fs.targets))
            target_selected = fs.targets[target_selected_option]

            col1, col2 = st.columns([1,3])
            with col1:
                corr_method = st.selectbox("相关性分析方法",["pearson","spearman","kendall"])
                correlation_threshold = st.slider("相关性阈值",0.001, 1.0, 0.9) # 0.52
                corr_matrix = pd.concat([fs.features, target_selected], axis=1).corr(corr_method)
                fs.identify_collinear(corr_matrix, correlation_threshold)
                fs.judge_drop_f_t_after_f_f([target_selected_option], corr_matrix)

                is_mask = st.selectbox('掩码',('Yes', 'No'))
                with st.expander('绘图参数'):
                    options_selected = [plot.set_tick_fontsize(21), plot.set_tick_fontsize(22)]
                with st.expander('丢弃的特征变量'):
                    st.write(fs.record_collinear)
            with col2:
                fs.features_dropped_collinear = fs.features.drop(columns=fs.ops['collinear'])
                assert fs.features_dropped_collinear.size != 0,'zero feature !' 
                corr_matrix_drop_collinear = fs.features_dropped_collinear.corr(corr_method)
                plot.corr_cofficient(options_selected, is_mask, corr_matrix_drop_collinear)
                with st.expander('处理之后的数据'):
                    data = pd.concat([fs.features_dropped_collinear, targets], axis=1)
                    st.write(data)
                    tmp_download_link = download_button(data, f'双线性特征变量处理数据.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)

    elif sub_option == "特征和目标相关性":
        colored_header(label="特征和目标相关性",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
        if file is None:
            table = PrettyTable(['上传文件名称', '名称','数据说明'])
            table.add_row(['file_1','dataset','数据集'])
            st.write(table)
        if file is not None:
            df = pd.read_csv(file)
            # 检测缺失值
            check_string_NaN(df)
            colored_header(label="数据信息", description=" ",color_name="violet-70")
            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

            target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)
            
            col_feature, col_target = st.columns(2)
            
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())
        
            colored_header(label="丢弃与目标的低相关性特征",description=" ",color_name="violet-70")
            fs = FeatureSelector(features, targets)
            plot = customPlot() 
            target_selected_option = st.selectbox('选择特征', list(fs.targets))
            col1, col2 = st.columns([1,3])
            
            with col1:  
                corr_method = st.selectbox("相关性分析方法",["pearson","spearman","kendall","MIR"], key=15)  
                if corr_method != "MIR":
                    option_dropped_threshold = st.slider('相关性阈值',0.0, 1.0,0.0)
                if corr_method == 'MIR':
                    options_seed = st.checkbox('random state 1024',True)
                with st.expander('绘图参数'):
                    options_selected = [plot.set_title_fontsize(11),plot.set_label_fontsize(12),
                        plot.set_tick_fontsize(13),plot.set_legend_fontsize(14),plot.set_color('bin color',19,16)]
                
            with col2:
                target_selected = fs.targets[target_selected_option]
                if corr_method != "MIR":
                    corr_matrix = pd.concat([fs.features, target_selected], axis=1).corr(corr_method).abs()

                    fs.judge_drop_f_t([target_selected_option], corr_matrix, option_dropped_threshold)
                    
                    fs.features_dropped_f_t = fs.features.drop(columns=fs.ops['f_t_low_corr'])
                    corr_f_t = pd.concat([fs.features_dropped_f_t, target_selected], axis=1).corr(corr_method)[target_selected_option][:-1]

                    plot.corr_feature_target(options_selected, corr_f_t)
                    with st.expander('处理之后的数据'):
                        data = pd.concat([fs.features_dropped_f_t, targets], axis=1)
                        st.write(data)
                        tmp_download_link = download_button(data, f'丢弃与目标的低相关性特征数据.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)
                else:
                    if options_seed:
                        corr_mir  = MIR(fs.features, target_selected, random_state=1024)
                    else:
                        corr_mir = MIR(fs.features, target_selected)
                    corr_mir = pd.DataFrame(corr_mir).set_index(pd.Index(list(fs.features.columns)))
                    corr_mir.rename(columns={0: 'mutual info'}, inplace=True)
                    plot.corr_feature_target_mir(options_selected, corr_mir)
            st.write('---')

    elif sub_option == "One-hot编码":
        colored_header(label="One-hot编码",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
        if file is None:
            table = PrettyTable(['上传文件名称', '名称','数据说明'])
            table.add_row(['file_1','dataset','数据集'])
            st.write(table)
        if file is not None:
            df = pd.read_csv(file)
            # 检测缺失值
            check_string_NaN(df)
            colored_header(label="数据信息", description=" ",color_name="violet-70")
            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

            target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)
            
            col_feature, col_target = st.columns(2)
            
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())
        
        #=============== drop major missing features ================
    
            colored_header(label="特征变量one-hot编码",description=" ",color_name="violet-70")
            fs = FeatureSelector(features, targets)
            plot = customPlot() 
            str_col_list = fs.features.select_dtypes(include=['object']).columns.tolist()
            fs.one_hot_feature_encoder(True)
            data = pd.concat([fs.features_plus_oneHot, targets], axis=1)
            # delete origin string columns
            data = data.drop(str_col_list, axis=1)
            st.write(data)
            tmp_download_link = download_button(data, f'特征变量one-hot编码数据.csv', button_text='download')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
            st.write('---')
    
    elif sub_option == "特征重要性":
        colored_header(label="特征重要性",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
        if file is None:
            table = PrettyTable(['上传文件名称', '名称','数据说明'])
            table.add_row(['file_1','dataset','数据集'])
            st.write(table)
        if file is not None:
            df = pd.read_csv(file)
            # 检测缺失值
            check_string_NaN(df)
            colored_header(label="数据信息", description=" ",color_name="violet-70")
            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

            target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)
            
            col_feature, col_target = st.columns(2)        
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())
    
            fs = FeatureSelector(features,targets)

            colored_header(label="选择目标变量", description=" ", color_name="violet-70")

            target_selected_name = st.selectbox('目标变量', list(fs.targets)[::-1])

            fs.targets = targets[target_selected_name]
            
            colored_header(label="Selector", description=" ",color_name="violet-70")

            model_path = './models/feature importance'
            
            template_alg = model_platform(model_path=model_path)

            colored_header(label="Training", description=" ",color_name="violet-70")

            inputs, col2 = template_alg.show()
            # st.write(inputs)

            if inputs['model'] == 'LinearRegressor':
                
                fs.model = LinearR()

                with col2:
                    option_cumulative_importance = st.slider('累计重要性阈值',0.0, 1.0, 0.95)
                    Embedded_method = st.checkbox('Embedded method',False)
                    if Embedded_method:
                        cv = st.number_input('cv',1,20,5)
                with st.container():
                    button_train = st.button('train', use_container_width=True)
                if button_train:
                    fs.LinearRegressor()     
                    fs.identify_zero_low_importance(option_cumulative_importance)
                    fs.feature_importance_select_show()
                    if Embedded_method:
                        threshold  = fs.cumulative_importance

                        feature_importances = fs.feature_importances.set_index('feature',drop = False)

                        features = []
                        scores = []
                        cumuImportance = []
                        for i in range(1, len(fs.features.columns) + 1):
                            features.append(feature_importances.iloc[:i, 0].values.tolist())
                            X_selected = fs.features[features[-1]]
                            score = CVS(fs.model, X_selected, fs.targets, cv=cv ,scoring='r2').mean()

                            cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                            scores.append(score)
                        cumu_importance = np.array(cumuImportance)
                        scores = np.array(scores) 
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            ax = plt.plot(cumu_importance, scores,'o-')
                            plt.xlabel("cumulative feature importance")
                            plt.ylabel("r2")
                            st.pyplot(fig)
            elif inputs['model'] == 'LassoRegressor':
                
                fs.model = Lasso(random_state=inputs['random state'])

                with col2:
                    option_cumulative_importance = st.slider('累计重要性阈值',0.0, 1.0, 0.95)
                    Embedded_method = st.checkbox('Embedded method',False)
                    if Embedded_method:
                        cv = st.number_input('cv',1,20,5)

                with st.container():
                    button_train = st.button('train', use_container_width=True)
                if button_train:

                    fs.LassoRegressor()

                    fs.identify_zero_low_importance(option_cumulative_importance)
                    fs.feature_importance_select_show()
                    if Embedded_method:
                        
                        threshold  = fs.cumulative_importance

                        feature_importances = fs.feature_importances.set_index('feature',drop = False)

                        features = []
                        scores = []
                        cumuImportance = []
                        for i in range(1, len(fs.features.columns) + 1):
                            features.append(feature_importances.iloc[:i, 0].values.tolist())
                            X_selected = fs.features[features[-1]]
                            score = CVS(fs.model, X_selected, fs.targets, cv=cv, scoring='r2').mean()

                            cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                            scores.append(score)
                        cumu_importance = np.array(cumuImportance)
                        scores = np.array(scores) 
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            ax = plt.plot(cumu_importance, scores,'o-')
                            plt.xlabel("cumulative feature importance")
                            plt.ylabel("r2")
                            st.pyplot(fig)

            elif inputs['model'] == 'RidgeRegressor':

                fs.model = Ridge(random_state=inputs['random state'])

                with col2:
                    option_cumulative_importance = st.slider('累计重要性阈值',0.0, 1.0, 0.95)
                    Embedded_method = st.checkbox('Embedded method',False)
                    if Embedded_method:
                        cv = st.number_input('cv',1,20,5)
                with st.container():
                    button_train = st.button('train', use_container_width=True)
                if button_train:
                    fs.RidgeRegressor()     
                    fs.identify_zero_low_importance(option_cumulative_importance)
                    fs.feature_importance_select_show()
                    if Embedded_method:
                        
                        threshold  = fs.cumulative_importance

                        feature_importances = fs.feature_importances.set_index('feature',drop = False)

                        features = []
                        scores = []
                        cumuImportance = []
                        for i in range(1, len(fs.features.columns) + 1):
                            features.append(feature_importances.iloc[:i, 0].values.tolist())
                            X_selected = fs.features[features[-1]]
                            score = CVS(fs.model, X_selected, fs.targets, cv=cv, scoring='r2').mean()

                            cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                            scores.append(score)
                        cumu_importance = np.array(cumuImportance)
                        scores = np.array(scores) 
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            ax = plt.plot(cumu_importance, scores,'o-')
                            plt.xlabel("cumulative feature importance")
                            plt.ylabel("r2")
                            st.pyplot(fig)
            elif inputs['model'] == 'LassoRegressor':
                
                fs.model = Lasso(random_state=inputs['random state'])

                with col2:
                    option_cumulative_importance = st.slider('累计重要性阈值',0.0, 1.0, 0.95)
                    Embedded_method = st.checkbox('Embedded method',False)
                    if Embedded_method:
                        cv = st.number_input('cv',1,20,5)
                with st.container():
                    button_train = st.button('train', use_container_width=True)
                if button_train:

                    fs.LassoRegressor()

                    fs.identify_zero_low_importance(option_cumulative_importance)
                    fs.feature_importance_select_show()
                    if Embedded_method:
                        
                        threshold  = fs.cumulative_importance

                        feature_importances = fs.feature_importances.set_index('feature',drop = False)

                        features = []
                        scores = []
                        cumuImportance = []
                        for i in range(1, len(fs.features.columns) + 1):
                            features.append(feature_importances.iloc[:i, 0].values.tolist())
                            X_selected = fs.features[features[-1]]
                            score = CVS(fs.model, X_selected, fs.targets, cv=cv, scoring='r2').mean()

                            cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                            scores.append(score)
                        cumu_importance = np.array(cumuImportance)
                        scores = np.array(scores) 
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            ax = plt.plot(cumu_importance, scores,'o-')
                            plt.xlabel("cumulative feature importance")
                            plt.ylabel("r2")
                            st.pyplot(fig)

            elif inputs['model'] == 'RandomForestRegressor':
                        
                        fs.model = RFR(criterion = inputs['criterion'], n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                        min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                                        n_jobs=inputs['njobs'])
                        with col2:
                            option_cumulative_importance = st.slider('累计重要性阈值',0.5, 1.0, 0.95)
                            Embedded_method = st.checkbox('Embedded method',False)
                            if Embedded_method:
                                cv = st.number_input('cv',1,20,5)
                            
                        with st.container():
                            button_train = st.button('train', use_container_width=True)
                        if button_train:

                            fs.RandomForestRegressor()

                            fs.identify_zero_low_importance(option_cumulative_importance)
                            fs.feature_importance_select_show()

                            if Embedded_method:
                                
                                threshold  = fs.cumulative_importance

                                feature_importances = fs.feature_importances.set_index('feature',drop = False)

                                features = []
                                scores = []
                                cumuImportance = []
                                for i in range(1, len(fs.features.columns) + 1):
                                    features.append(feature_importances.iloc[:i, 0].values.tolist())
                                    X_selected = fs.features[features[-1]]
                                    score = CVS(fs.model, X_selected, fs.targets, cv=cv ,scoring='r2').mean()

                                    cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                                    scores.append(score)
                                cumu_importance = np.array(cumuImportance)
                                scores = np.array(scores) 
                                with plt.style.context(['nature','no-latex']):
                                    fig, ax = plt.subplots()
                                    ax = plt.plot(cumu_importance, scores,'o-')
                                    plt.xlabel("cumulative feature importance")
                                    plt.ylabel("r2")
                                    st.pyplot(fig)

            st.write('---')

elif select_option == "回归预测":

    colored_header(label="回归预测",description=" ",color_name="violet-90")
    file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
    if file is None:
        table = PrettyTable(['上传文件名称', '名称','数据说明'])
        table.add_row(['file_1','dataset','数据集'])
        st.write(table)
    if file is not None:
        df = pd.read_csv(file)
        # 检测缺失值
        check_string_NaN(df)

        colored_header(label="数据信息", description=" ",color_name="violet-70")
        nrow = st.slider("rows", 1, len(df), 5)
        df_nrow = df.head(nrow)
        st.write(df_nrow)

        colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

        target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)
        
        col_feature, col_target = st.columns(2)
        # features
        features = df.iloc[:,:-target_num]
        # targets
        targets = df.iloc[:,-target_num:]
        with col_feature:    
            st.write(features.head())
        with col_target:   
            st.write(targets.head())
# =================== model ====================================
        reg = REGRESSOR(features,targets)

        colored_header(label="选择目标变量", description=" ", color_name="violet-70")

        target_selected_option = st.selectbox('target', list(reg.targets)[::-1])

        reg.targets = targets[target_selected_option]

        colored_header(label="Regressor", description=" ",color_name="violet-30")

        model_path = './models/regressors'

        template_alg = model_platform(model_path)

        inputs, col2 = template_alg.show()
    
        if inputs['model'] == 'DecisionTreeRegressor':

            with col2:
                with st.expander('Operator'):
                    operator = st.selectbox('', ('train test split','cross val score', 'leave one out'), label_visibility= "collapsed")
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])

                    elif operator == 'cross val score':
                        cv = st.number_input('cv',1,20,5)
                    
                    elif operator == 'leave one out':
                        loo = LeaveOneOut()
            
            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                if operator == 'train test split':

                    reg.model = tree.DecisionTreeRegressor(random_state=inputs['random state'],splitter=inputs['splitter'],
                                    max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                    min_samples_split=inputs['min samples split']) 
                    
                    reg.DecisionTreeRegressor()

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    plot_and_export_results(reg, "DTR")

                    if inputs['tree graph']:
                        class_names = list(set(reg.targets.astype(str).tolist()))
                        dot_data = tree.export_graphviz(reg.model,out_file=None, feature_names=list(reg.features), class_names=class_names,filled=True, rounded=True)
                        graph = graphviz.Source(dot_data)
                        graph.render('Tree graph', view=True)

                elif operator == 'cross val score':

                    reg.model = tree.DecisionTreeRegressor(random_state=inputs['random state'],splitter=inputs['splitter'],
                        max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                        min_samples_split=inputs['min samples split']) 

                    export_cross_val_results(reg, cv, "DTR_cv", inputs['random state'])
        
                elif operator == 'leave one out':

                    reg.model = tree.DecisionTreeRegressor(random_state=inputs['random state'],splitter=inputs['splitter'],
                        max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                        min_samples_split=inputs['min samples split']) 
                    
                    export_loo_results(reg, loo, "DTR_loo")

        if inputs['model'] == 'RandomForestRegressor':
            with col2:
                with st.expander('Operator'):
                    operator = st.selectbox('data operator', ('train test split','cross val score','leave one out','oob score'))
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                    elif operator == 'cross val score':
                        cv = st.number_input('cv',1,20,5)

                    elif operator == 'leave one out':
                        loo = LeaveOneOut()

                    elif operator == 'oob score':
                        inputs['oob score']  = st.selectbox('oob score',[True], disabled=True)
                        inputs['warm start'] = True

            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            
            if button_train:
                if operator == 'train test split':

                    reg.model = RFR( n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                    min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                                    n_jobs=inputs['njobs'])
                    
                    reg.RandomForestRegressor()

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    plot_and_export_results(reg, "RFR")


                elif operator == 'cross val score':

                    reg.model = RFR(n_estimators=inputs['nestimators'],random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                                n_jobs=inputs['njobs'])

                    export_cross_val_results(reg, cv, "RFR_cv", inputs['random state'])
    
                elif operator == 'oob score':

                    reg.model = RFR(criterion = inputs['criterion'],n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                                n_jobs=inputs['njobs'])
                
                    reg_res  = reg.model.fit(reg.features, reg.targets)
                    oob_score = reg_res.oob_score_
                    st.write(f'oob score : {oob_score}')

                elif operator == 'leave one out':

                    reg.model = RFR(criterion = inputs['criterion'],n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                                n_jobs=inputs['njobs'])
                    export_loo_results(reg, loo, "RFR_loo")

        if inputs['model'] == 'SupportVector':

            with col2:
                with st.expander('Operator'):

                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])

                    operator = st.selectbox('operator', ('train test split','cross val score', 'leave one out'), label_visibility='collapsed')
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        
                        reg.features = pd.DataFrame(reg.features)    
                        
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])

                    elif operator == 'cross val score':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('cv',1,20,5)

                    elif operator == 'leave one out':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()
            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                if operator == 'train test split':

                    reg.model = SVR(kernel=inputs['kernel'], C=inputs['C'])
                    
                    reg.SupportVector()

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    
                    plot_and_export_results(reg, "SVR")

                elif operator == 'cross val score':

                    reg.model = SVR(kernel=inputs['kernel'], C=inputs['C'])

                    export_cross_val_results(reg, cv, "SVR_cv",inputs['random state'])


                elif operator == 'leave one out':

                    reg.model = SVR(kernel=inputs['kernel'], C=inputs['C'])
                
                    export_loo_results(reg, loo, "SVR_loo")

        if inputs['model'] == 'GPRegressor':

            with col2:
                with st.expander('Operator'):

                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])

                    operator = st.selectbox('operator', ('train test split','cross val score', 'leave one out'), label_visibility='collapsed')
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        
                        reg.features = pd.DataFrame(reg.features)    
                        
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])

                    elif operator == 'cross val score':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('cv',1,20,5)

                    elif operator == 'leave one out':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()
            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                if operator == 'train test split':
                    if inputs['kernel'] == None:
                        kernel = None
                    elif inputs['kernel'] == 'DotProduct':
                        kernel = DotProduct()
                    elif inputs['kernel'] == 'WhiteKernel':
                        kernel = WhiteKernel()
                    else:
                        kernel = DotProduct() + WhiteKernel()
                    reg.model = GPR(kernel = kernel, random_state = inputs['random state'])
                    
                    reg.GPRegressor()

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    
                    plot_and_export_results(reg, "GPR")

                elif operator == 'cross val score':
                    if inputs['kernel'] == None:
                        kernel = None
                    elif inputs['kernel'] == 'DotProduct':
                        kernel = DotProduct()
                    elif inputs['kernel'] == 'WhiteKernel':
                        kernel = WhiteKernel()
                    else:
                        kernel = DotProduct() + WhiteKernel()
                    reg.model = GPR(kernel = kernel, random_state = inputs['random state'])

                    export_cross_val_results(reg, cv, "GPR_cv",inputs['random state'])

                elif operator == 'leave one out':
                    if inputs['kernel'] == None:
                        kernel = None
                    elif inputs['kernel'] == 'DotProduct':
                        kernel = DotProduct()
                    elif inputs['kernel'] == 'WhiteKernel':
                        kernel = WhiteKernel()
                    else:
                        kernel = DotProduct() + WhiteKernel()
                    reg.model = GPR(kernel = kernel, random_state = inputs['random state'])
                
                    export_loo_results(reg, loo, "GPR_loo")

        if inputs['model'] == 'KNeighborsRegressor':

            with col2:
                with st.expander('Operator'):

                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])

                    operator = st.selectbox('operator', ('train test split','cross val score', 'leave one out'), label_visibility='collapsed')
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        
                        reg.features = pd.DataFrame(reg.features)    
                        
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif operator == 'cross val score':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('cv',1,20,5)

                    elif operator == 'leave one out':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()

            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                if operator == 'train test split':

                    reg.model = KNeighborsRegressor(n_neighbors = inputs['n neighbors'])
                    
                    reg.KNeighborsRegressor()

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    
                    plot_and_export_results(reg, "KNR")

                elif operator == 'cross val score':

                    reg.model = KNeighborsRegressor(n_neighbors = inputs['n neighbors'])

                    export_cross_val_results(reg, cv, "KNR_cv", inputs['random state'])

                elif operator == 'leave one out':

                    reg.model = KNeighborsRegressor(n_neighbors = inputs['n neighbors'])
                
                    export_loo_results(reg, loo, "KNR_loo")

        if inputs['model'] == 'LinearRegressor':

            with col2:
                with st.expander('Operator'):

                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])

                    operator = st.selectbox('operator', ('train test split','cross val score', 'leave one out'), label_visibility='collapsed')
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        
                        reg.features = pd.DataFrame(reg.features)    
                        
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif operator == 'cross val score':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('cv',1,20,5)
    
                    elif operator == 'leave one out':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()

            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                if operator == 'train test split':

                    reg.model = LinearR()
                    
                    reg.LinearRegressor()
                    # plot = customPlot()
                    # plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    plot_and_export_results(reg, "LinearR")                     


                elif operator == 'cross val score':

                    reg.model = LinearR()

                    export_cross_val_results(reg, cv, "LinearR_cv" ,inputs['random state'])


                elif operator == 'leave one out':

                    reg.model = LinearR()
                    
                    export_loo_results(reg, loo, "LinearR_loo")
                                
        if inputs['model'] == 'LassoRegressor':

            with col2:
                with st.expander('Operator'):

                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])

                    operator = st.selectbox('', ('train test split','cross val score', 'leave one out'), label_visibility='collapsed')
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        
                        reg.features = pd.DataFrame(reg.features)    
                        
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif operator == 'cross val score':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('cv',1,20,5)

                    elif operator == 'leave one out':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()
            
            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                if operator == 'train test split':

                    reg.model = Lasso(alpha=inputs['alpha'],random_state=inputs['random state'])
                    
                    reg.LassoRegressor()

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    
                    plot_and_export_results(reg, "LassoR")

                elif operator == 'cross val score':

                    reg.model = Lasso(alpha=inputs['alpha'],random_state=inputs['random state'])

                    export_cross_val_results(reg, cv, "LassoR_cv", inputs['random state'])

                elif operator == 'leave one out':

                    reg.model = Lasso(alpha=inputs['alpha'],random_state=inputs['random state'])
                
                    export_loo_results(reg, loo, "LassoR_loo")

        if inputs['model'] == 'RidgeRegressor':

            with col2:
                with st.expander('Operator'):

                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])

                    operator = st.selectbox('data operator', ('train test split','cross val score', 'leave one out'))
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        
                        reg.features = pd.DataFrame(reg.features)    
                        
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif operator == 'cross val score':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('cv',1,20,5)
                    
                    elif operator == 'leave one out':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()              
            
            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                if operator == 'train test split':

                    reg.model = Ridge(alpha=inputs['alpha'], random_state=inputs['random state'])
                    
                    reg.RidgeRegressor()

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']

                    plot_and_export_results(reg, "RidgeR")

                elif operator == 'cross val score':

                    reg.model = Ridge(alpha=inputs['alpha'], random_state=inputs['random state'])

                    export_cross_val_results(reg, cv, "RidgeR_cv", inputs['random state'])


                elif operator == 'leave one out':

                    reg.model = Ridge(alpha=inputs['alpha'], random_state=inputs['random state'])
                
                    export_loo_results(reg, loo, "RidgeR_loo")

        if inputs['model'] == 'BaggingRegressor':
            with col2:
                with st.expander('Operator'):
                    # preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])

                    operator = st.selectbox('operator', ('train test split','cross val score', 'leave one out'), label_visibility='collapsed')
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        # if preprocess == 'StandardScaler':
                        #     reg.features = StandardScaler().fit_transform(reg.features)
                        # if preprocess == 'MinMaxScaler':
                        #     reg.features = MinMaxScaler().fit_transform(reg.features)
                        
                        reg.features = pd.DataFrame(reg.features)    
                        
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])

                    elif operator == 'cross val score':
                        # if preprocess == 'StandardScaler':
                        #     reg.features = StandardScaler().fit_transform(reg.features)
                        # if preprocess == 'MinMaxScaler':
                        #     reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('cv',1,20,5)

                    elif operator == 'leave one out':
                        # if preprocess == 'StandardScaler':
                        #     reg.features = StandardScaler().fit_transform(reg.features)
                        # if preprocess == 'MinMaxScaler':
                        #     reg.features = MinMaxScaler().fit_transform(reg.features)
                        reg.features = pd.DataFrame(reg.features)    
                        loo = LeaveOneOut()

            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:

                if operator == 'train test split':

                # if inputs['base estimator'] == "DecisionTree": 
                    reg.model = BaggingRegressor(estimator = tree.DecisionTreeRegressor(random_state=inputs['random state']),n_estimators=inputs['nestimators'],
                                                    max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
                    
                    reg.BaggingRegressor()
    

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']

                    plot_and_export_results(reg, "BaggingR")

                    
                    # elif inputs['base estimator'] == "SupportVector": 
                    #     reg.model = BaggingRegressor(estimator = SVR(),n_estimators=inputs['nestimators'],
                    #             max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
                    #     reg.BaggingRegressor()

                    #     result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    #     result_data.columns = ['actual','prediction']

                    #     export_cross_val_results(reg, cv, "DTR_cv")
                    
                    # elif inputs['base estimator'] == "LinearRegression": 
                    #     reg.model = BaggingRegressor(estimator = LinearR(),n_estimators=inputs['nestimators'],
                    #             max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
                    #     reg.BaggingRegressor()
        

                    #     result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    #     result_data.columns = ['actual','prediction']

                    #     plot_and_export_results(reg, "BaggingR")

                elif operator == 'cross val score':
                # if inputs['base estimator'] == "DecisionTree": 
                    reg.model = BaggingRegressor(estimator = tree.DecisionTreeRegressor(random_state=inputs['random state']),n_estimators=inputs['nestimators'],
                                                    max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 

                    # cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)

                    # export_cross_val_results(cvs, "BaggingR_cv")   
                    export_cross_val_results(reg, cv, "BaggingR_cv", inputs['random state'])

                    # elif inputs['base estimator'] == "SupportVector": 

                    #     reg.model = BaggingRegressor(estimator =  SVR(),n_estimators=inputs['nestimators'],
                    #             max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 

                    #     cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)
            
                    #     export_cross_val_results(cvs, "BaggingR_cv")   

                    # elif inputs['base estimator'] == "LinearRegression":
                    #     reg.model = BaggingRegressor(estimator = LinearR(),n_estimators=inputs['nestimators'],
                    #             max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 

                    #     cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)
            
                    #     export_cross_val_results(cvs, "BaggingR_cv")   

                elif operator == 'leave one out':
                # if inputs['base estimator'] == "DecisionTree": 
                    reg.model = BaggingRegressor(estimator = tree.DecisionTreeRegressor(random_state=inputs['random state']),n_estimators=inputs['nestimators'],
                            max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
                
                    export_loo_results(reg, loo, "BaggingR_loo")

                    # elif inputs['base estimator'] == "SupportVector": 

                    #     reg.model = BaggingRegressor(estimator = SVR(),n_estimators=inputs['nestimators'],
                    #             max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
                        
                    #     export_loo_results(reg, loo, "BaggingR_loo")

                    # elif inputs['base estimator'] == "LinearRegression":
                    #     reg.model = BaggingRegressor(estimator = LinearR(),n_estimators=inputs['nestimators'],
                    #             max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
        
                    #     export_loo_results(reg, loo, "BaggingR_loo")

        if inputs['model'] == 'AdaBoostRegressor':

            with col2:
                with st.expander('Operator'):
                    operator = st.selectbox('data operator', ('train test split','cross val score','leave one out'))
                
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                    elif operator == 'cross val score':
                        cv = st.number_input('cv',1,20,5)
                    elif operator == 'leave one out':
                        loo = LeaveOneOut()
            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:

                if operator == 'train test split':

                    # if inputs['base estimator'] == "DecisionTree": 
                    reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 
                    
                    reg.AdaBoostRegressor()
    
                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']

                    plot_and_export_results(reg, "AdaBoostR")
                    
                    # elif inputs['base estimator'] == "SupportVector": 

                    #     reg.model = AdaBoostRegressor(estimator=SVR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state'])
                        
                    #     reg.AdaBoostRegressor()

                    #     result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    #     result_data.columns = ['actual','prediction']

                    #     plot_and_export_results(reg, "AdaBoostR")
                    
                    # elif inputs['base estimator'] == "LinearRegression": 
                    #     reg.model = AdaBoostRegressor(estimator=LinearR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state'])
                    #     reg.AdaBoostRegressor()

                    #     result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    #     result_data.columns = ['actual','prediction']
                        
                    #     plot_and_export_results(reg, "AdaBoostR")

                elif operator == 'cross val score':
                    # if inputs['base estimator'] == "DecisionTree": 
                    reg.model =  AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 
                    # cvs = CVS(reg.model, reg.features, reg.targets, cv = cv)
                    export_cross_val_results(reg, cv, "AdaBoostR_cv", inputs['random state'])
                    #     st.write('mean cross val R2:', cvs.mean())
                    # elif inputs['base estimator'] == "SupportVector": 

                    #     reg.model =  AdaBoostRegressor(estimator=SVR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 

                    #     cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)
            
                    #     export_cross_val_results(cvs, "AdaBoostR_cv")  

                    # elif inputs['base estimator'] == "LinearRegression":
                    #     reg.model =  AdaBoostRegressor(estimator=LinearR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 

                    #     cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)
            
                    #     export_cross_val_results(cvs, "AdaBoostR_cv")  

                elif operator == 'leave one out':
                    # if inputs['base estimator'] == "DecisionTree": 
                    reg.model =  AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 
                
                    export_loo_results(reg, loo, "AdaBoostR_loo")
                    
                    # elif inputs['base estimator'] == "SupportVector": 

                    #     reg.model = reg.model = AdaBoostRegressor(estimator=SVR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 
                        
                    #     export_loo_results(reg, loo, "AdaBoostR_loo")

                    # elif inputs['base estimator'] == "LinearRegression":
                    #     reg.model = reg.model =  AdaBoostRegressor(estimator=LinearR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 
                                            
                    #     export_loo_results(reg, loo, "AdaBoostR_loo")
                    
        if inputs['model'] == 'GradientBoostingRegressor':

            with col2:
                with st.expander('Operator'):
                    operator = st.selectbox('data operator', ('train test split','cross val score','leave one out'))
                
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                    elif operator == 'cross val score':
                        cv = st.number_input('cv',1,20,5)
                    elif operator == 'leave one out':
                        loo = LeaveOneOut()

            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:

                if operator == 'train test split':

                        reg.model = GradientBoostingRegressor(learning_rate=inputs['learning rate'],n_estimators=inputs['nestimators'],max_features=inputs['max features'],
                                                            random_state=inputs['random state']) 
                        
                        reg.GradientBoostingRegressor()
        

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        plot_and_export_results(reg, "GradientBoostingR")

                elif operator == 'cross val score':
                        reg.model = GradientBoostingRegressor(learning_rate=inputs['learning rate'],n_estimators=inputs['nestimators'],max_features=inputs['max features'],
                                                            random_state=inputs['random state'])  
                        # cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)
                        # export_cross_val_results(cvs, "GradientBoostingR_cv")    
                        export_cross_val_results(reg, cv, "GradientBoostingR_cv", inputs['random state'])
                elif operator == 'leave one out':
                        reg.model = GradientBoostingRegressor(learning_rate=inputs['learning rate'],n_estimators=inputs['nestimators'],max_features=inputs['max features'],
                                                            random_state=inputs['random state']) 
                    
                        export_loo_results(reg, loo, "GradientBoostingR_loo")

        if inputs['model'] == 'XGBRegressor':

            with col2:
                with st.expander('Operator'):
                    operator = st.selectbox('data operator', ('train test split','cross val score','leave one out'))
                
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                    elif operator == 'cross val score':
                        cv = st.number_input('cv',1,20,5)
                    elif operator == 'leave one out':
                        loo = LeaveOneOut()
            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:

                if operator == 'train test split':

                        reg.model = xgb.XGBRegressor(booster=inputs['base estimator'], n_estimators=inputs['nestimators'], 
                                                    max_depth= inputs['max depth'], subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                    learning_rate=inputs['learning rate'])
                        reg.XGBRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']

                        plot_and_export_results(reg, "XGBR")

                elif operator == 'cross val score':
                        reg.model = xgb.XGBRegressor(booster=inputs['base estimator'], n_estimators=inputs['nestimators'], 
                                                    max_depth= inputs['max depth'], subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                    learning_rate=inputs['learning rate'])
                        
                        # cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)
                        # export_cross_val_results(cvs, "DTR_cv")    
                        export_cross_val_results(reg, cv, "XGBR_cv", inputs['random state'])

                elif operator == 'leave one out':
                        reg.model = xgb.XGBRegressor(booster=inputs['base estimator'], n_estimators=inputs['nestimators'], 
                                                    max_depth= inputs['max depth'], subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                    learning_rate=inputs['learning rate'])
                    
                        export_loo_results(reg, loo, "XGBR_loo")

        if inputs['model'] == 'CatBoostRegressor':
            with col2:
                with st.expander('Operator'):
                    operator = st.selectbox('data operator', ('train test split','cross val score','leave one out'))
                
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                    elif operator == 'cross val score':
                        cv = st.number_input('cv',1,20,5)
                    elif operator == 'leave one out':
                        loo = LeaveOneOut()
            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:

                if operator == 'train test split':

                        reg.model =CatBoostRegressor(iterations=inputs['niteration'],learning_rate=inputs['learning rate'],depth = inputs['max depth'])

                        reg.LGBMRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']

                        plot_and_export_results(reg, "CatBoostR")

                elif operator == 'cross val score':
                        reg.model = CatBoostRegressor(iterations=inputs['niteration'],learning_rate=inputs['learning rate'],depth = inputs['max depth'])
                        
                        # cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)
                        # export_cross_val_results(cvs, "CatBoostR_cv")    
                        export_cross_val_results(reg, cv, "CatBoostR_cv", inputs['random state'])

                elif operator == 'leave one out':
                        reg.model = CatBoostRegressor(iterations=inputs['niteration'],learning_rate=inputs['learning rate'],depth = inputs['max depth'])
                        export_loo_results(reg, loo, "CatBoostR_loo")            
        
        if inputs['model'] == 'MLPRegressor':
            with col2:
                with st.expander('Operator'):

                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])

                    operator = st.selectbox('operator', ('train test split','cross val score', 'leave one out'), label_visibility='collapsed')
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        
                        reg.features = pd.DataFrame(reg.features)    
                        
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif operator == 'cross val score':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('cv',1,20,5)
                    
                    elif operator == 'leave one out':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()              
            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                if operator == 'train test split':

                    reg.model = MLPRegressor(hidden_layer_sizes=inputs['hidden layer size'], activation= inputs['activation'], solver=inputs['solver'], 
                                            batch_size=inputs['batch size'], learning_rate= inputs['learning rate'], max_iter=inputs['max iter'],
                                            random_state=inputs['random state'])
                    reg.MLPRegressor()

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    plot_and_export_results(reg, "MLP")

                elif operator == 'cross val score':

                    reg.model = MLPRegressor(hidden_layer_sizes=inputs['hidden layer size'], activation= inputs['activation'], solver=inputs['solver'], 
                                            batch_size=inputs['batch size'], learning_rate= inputs['learning rate'], max_iter=inputs['max iter'],
                                            random_state=inputs['random state'])
                    
                    export_cross_val_results(reg, cv, "MLP_cv", inputs['random state'])

                elif operator == 'leave one out':

                    reg.model = MLPRegressor(hidden_layer_sizes=inputs['hidden layer size'], activation= inputs['activation'], solver=inputs['solver'], 
                                            batch_size=inputs['batch size'], learning_rate= inputs['learning rate'], max_iter=inputs['max iter'],
                                            random_state=inputs['random state'])
                
                    export_loo_results(reg, loo, "MLP_loo")

            st.write('---')
                            
elif select_option == "分类预测":

    colored_header(label="分类预测",description=" ",color_name="violet-90")
    file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
    if file is None:
        table = PrettyTable(['上传文件名称', '名称','数据说明'])
        table.add_row(['file_1','dataset','数据集'])
        st.write(table)
    if file is not None:
        df = pd.read_csv(file)
        check_string(df)
        colored_header(label="数据信息", description=" ",color_name="violet-70")
        nrow = st.slider("rows", 1, len(df), 5)
        df_nrow = df.head(nrow)
        st.write(df_nrow)

        colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

        target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)
        
        col_feature, col_target = st.columns(2)
            
        # features
        features = df.iloc[:,:-target_num]
        # targets
        targets = df.iloc[:,-target_num:]
        with col_feature:    
            st.write(features.head())
        with col_target:   
            st.write(targets.head())

        clf = CLASSIFIER(features,targets)

        colored_header(label="Choose Target", description=" ", color_name="violet-30")
        target_selected_option = st.selectbox('target', list(clf.targets)[::-1])

        clf.targets = pd.DataFrame(targets[target_selected_option])
        # col_name = []
        # if check_string(clf.targets):
        col_name = list(clf.targets)
        clf.targets[col_name[0]], unique_categories = pd.factorize(clf.targets[col_name[0]])
        # st.write(fs.targets.head())
        colored_header(label="Classifier", description=" ",color_name="violet-30")

        model_path = './models/classifiers'
        
        template_alg = model_platform(model_path)

        colored_header(label="Training", description=" ",color_name="violet-30")

        inputs, col2 = template_alg.show()
       
        if inputs['model'] == 'DecisionTreeClassifier':

            with col2:
                with st.expander('Operator'):
                    data_process = st.selectbox('data process', ('train test split','cross val score','leave one out'), label_visibility='collapsed')
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                    elif data_process == 'cross val score':
                        cv = st.number_input('cv',1,20,5)
                    elif data_process == 'leave one out':
                        loo = LeaveOneOut()

            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                if data_process == 'train test split':
                            
                    clf.model = tree.DecisionTreeClassifier(criterion=inputs['criterion'],random_state=inputs['random state'],splitter=inputs['splitter'],
                                                            max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],min_samples_split=inputs['min samples split'])
                                                               
                    clf.DecisionTreeClassifier()

                    if not check_string(targets):
                        conf_matrix = confusion_matrix(clf.Ytest.values, clf.Ypred.values)
                        conf_df = pd.DataFrame(conf_matrix, index=np.unique(clf.Ytest))
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, fmt="d",cmap='Blues')
                            plt.xlabel('Actual')
                            plt.ylabel('Prediction')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)
                    else:
                        clf.Ytest[col_name[0]] = pd.Series(unique_categories).iloc[clf.Ytest[col_name[0]]].values
                        clf.Ypred[col_name[0]] = pd.Series(unique_categories).iloc[clf.Ypred[col_name[0]]].values
                        conf_matrix = confusion_matrix(clf.Ytest.values, clf.Ypred.values, labels=np.unique(clf.Ytest))
                        conf_df = pd.DataFrame(conf_matrix, index=np.unique(clf.Ytest), columns=np.unique(clf.Ytest))
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, fmt="d",cmap='Blues',
                                        xticklabels=conf_df.columns, yticklabels=conf_df.index)
                            plt.xlabel('Actual')
                            plt.ylabel('Prediction')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)

                    result_data = pd.concat([clf.Ytest, clf.Ypred], axis=1)
                    result_data.columns = ['actual','prediction']
                    with st.expander('Actual vs Predict'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'actual vs prediciton.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)
                    if inputs['tree graph']:
                        class_names = list(set(clf.targets.astype(str).tolist()))
                        dot_data = tree.export_graphviz(clf.model,out_file=None, feature_names=list(clf.features), class_names=class_names,filled=True, rounded=True)
                        graph = graphviz.Source(dot_data)
                        graph.render('Tree graph', view=True)
                elif data_process == 'cross val score':
                    clf.model = tree.DecisionTreeClassifier(criterion=inputs['criterion'],random_state=inputs['random state'],splitter=inputs['splitter'],
                                                            max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],min_samples_split=inputs['min samples split'])
                                                            
                    cvs = CVS(clf.model, clf.features, clf.targets, cv = cv)
                    export_cross_val_results_clf(clf, cv, "DTC_cv", col_name, unique_categories, inputs['random state'])

                elif data_process == 'leave one out':
                    clf.model = tree.DecisionTreeClassifier(criterion=inputs['criterion'],random_state=inputs['random state'],splitter=inputs['splitter'],
                                                            max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],min_samples_split=inputs['min samples split'])                        
                    
                    export_loo_results_clf(clf, loo, "DTC_loo", col_name, unique_categories)

        if inputs['model'] == 'RandomForestClassifier':
      
            with col2:
                with st.expander('Operator'):
                    data_process = st.selectbox('data process', ('train test split','cross val score','leave one out'), label_visibility='collapsed')
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                    elif data_process == 'cross val score':
                        cv = st.number_input('cv',1,20,5)       
                    elif data_process == 'leave one out':
                        loo = LeaveOneOut() 

            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:

                if data_process == 'train test split':

                    clf.model = RFC(criterion = inputs['criterion'],n_estimators=inputs['nestimators'],random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                min_samples_split=inputs['min samples split'], oob_score=inputs['oob score'], warm_start=inputs['warm start'])
                    
                    clf.RandomForestClassifier()
                    
                    if not check_string(targets):
                        conf_matrix = confusion_matrix(clf.Ytest.values, clf.Ypred.values)
                        conf_df = pd.DataFrame(conf_matrix, index=np.unique(clf.Ytest))
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, fmt="d",cmap='Blues')
                            plt.xlabel('Actual')
                            plt.ylabel('Prediction')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)
                    else:
                        clf.Ytest[col_name[0]] = pd.Series(unique_categories).iloc[clf.Ytest[col_name[0]]].values
                        clf.Ypred[col_name[0]] = pd.Series(unique_categories).iloc[clf.Ypred[col_name[0]]].values
                        conf_matrix = confusion_matrix(clf.Ytest.values, clf.Ypred.values, labels=np.unique(clf.Ytest))
                        conf_df = pd.DataFrame(conf_matrix, index=np.unique(clf.Ytest), columns=np.unique(clf.Ytest))
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, cmap='Blues', fmt='d',
                                        xticklabels=conf_df.columns, yticklabels=conf_df.index)
                            plt.xlabel('Actual')
                            plt.ylabel('Prediction')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)
                    result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    with st.expander('Actual vs Predict'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'actual vs prediction.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)        
                elif data_process == 'cross val score':

                    clf.model = RFC(criterion = inputs['criterion'],n_estimators=inputs['nestimators'],random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                min_samples_split=inputs['min samples split'], oob_score=inputs['oob score'], warm_start=inputs['warm start'])
                
     
                    cvs = CVS(clf.model, clf.features, clf.targets, cv = cv)
     
                    export_cross_val_results_clf(clf, cv, "RFRC_cv", col_name, unique_categories, inputs['random state'])

                elif data_process == 'leave one out':
 
                    clf.model = RFC(criterion = inputs['criterion'],n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                min_samples_split=inputs['min samples split'], oob_score=inputs['oob score'], warm_start=inputs['warm start'])
                
                    export_loo_results_clf(clf, loo, "RFRC_loo", col_name, unique_categories)     

        if inputs['model'] == 'LogisticRegression':

            with col2:
                with st.expander('Operator'):
                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])
                    
                    data_process = st.selectbox('data process', ('train test split','cross val score','leave one out'), label_visibility='collapsed')
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif data_process == 'cross val score':

                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        cv = st.number_input('cv',1,20,5)

                    elif data_process == 'leave one out':

                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        loo = LeaveOneOut() 

            with st.container():
                button_train = st.button('Train', use_container_width=True)
            
            if button_train:
                if data_process == 'train test split':
                            
                    clf.model = LR(penalty=inputs['penalty'],C=inputs['C'],solver=inputs['solver'],max_iter=inputs['max iter'],multi_class=inputs['multi class'],
                                   random_state=inputs['random state'],l1_ratio= inputs['l1 ratio'])   
                    clf.LogisticRegreesion()
                    
                    if not check_string(targets):
                        conf_matrix = confusion_matrix(clf.Ytest.values, clf.Ypred.values)
                        conf_df = pd.DataFrame(conf_matrix, index=np.unique(clf.Ytest))
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, fmt="d",cmap='Blues')
                            plt.xlabel('Actual')
                            plt.ylabel('Prediction')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)
                    else:
                        clf.Ytest[col_name[0]] = pd.Series(unique_categories).iloc[clf.Ytest[col_name[0]]].values
                        clf.Ypred[col_name[0]] = pd.Series(unique_categories).iloc[clf.Ypred[col_name[0]]].values
                        conf_matrix = confusion_matrix(clf.Ytest.values, clf.Ypred.values, labels=np.unique(clf.Ytest))
                        conf_df = pd.DataFrame(conf_matrix, index=np.unique(clf.Ytest), columns=np.unique(clf.Ytest))
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, cmap='Blues',fmt='d',
                                        xticklabels=conf_df.columns, yticklabels=conf_df.index)
                            plt.xlabel('Actual')
                            plt.ylabel('Prediction')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)

                    result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    with st.expander('Actual vs Predict'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'actual vs prediction.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)
                elif data_process == 'cross val score':
                    clf.model = LR(penalty=inputs['penalty'],C=inputs['C'],solver=inputs['solver'],max_iter=inputs['max iter'],multi_class=inputs['multi class'],
                                   random_state=inputs['random state'],l1_ratio= inputs['l1 ratio'])   
                     
                    cvs = CVS(clf.model, clf.features, clf.targets, cv = cv)

                    export_cross_val_results_clf(clf, cv, "LRC_cv", col_name, unique_categories, inputs['random state'])      

                elif data_process == 'leave one out': 
                    clf.model = LR(penalty=inputs['penalty'],C=inputs['C'],solver=inputs['solver'],max_iter=inputs['max iter'],multi_class=inputs['multi class'],
                                   random_state=inputs['random state'],l1_ratio= inputs['l1 ratio']) 
 
                    export_loo_results_clf(clf, loo, "LRC_loo", col_name, unique_categories)          
        if inputs['model'] == 'SupportVector':
            with col2:
                with st.expander('Operator'):
                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])

                    data_process = st.selectbox('data process', ('train test split','cross val score', 'leave one out'), label_visibility='collapsed')
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])

                    elif data_process == 'cross val score':
                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        cv = st.number_input('cv',1,20,5)

                    elif data_process == 'leave one out':
                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        loo = LeaveOneOut() 

            with st.container():
                button_train = st.button('Train', use_container_width=True)   
            if button_train:
                if data_process == 'train test split':
                            
                    clf.model = SVC(C=inputs['C'], kernel=inputs['kernel'], class_weight=inputs['class weight'])
                                                               
                    clf.SupportVector()
                    
                    if not check_string(targets):
                        conf_matrix = confusion_matrix(clf.Ytest.values, clf.Ypred.values)
                        conf_df = pd.DataFrame(conf_matrix, index=np.unique(clf.Ytest))
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, fmt="d",cmap='Blues')
                            plt.xlabel('Actual')
                            plt.ylabel('Prediction')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)
                    else:
                        clf.Ytest[col_name[0]] = pd.Series(unique_categories).iloc[clf.Ytest[col_name[0]]].values
                        clf.Ypred[col_name[0]] = pd.Series(unique_categories).iloc[clf.Ypred[col_name[0]]].values
                        conf_matrix = confusion_matrix(clf.Ytest.values, clf.Ypred.values, labels=np.unique(clf.Ytest))
                        conf_df = pd.DataFrame(conf_matrix, index=np.unique(clf.Ytest), columns=np.unique(clf.Ytest))
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, cmap='Blues',fmt='d',
                                        xticklabels=conf_df.columns, yticklabels=conf_df.index)
                            plt.xlabel('Actual')
                            plt.ylabel('Prediction')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)

                    result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    with st.expander('Actual vs Predict'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'actual vs prediction.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif data_process == 'cross val score':
                    clf.model = SVC(C=inputs['C'], kernel=inputs['kernel'], class_weight=inputs['class weight'])                                                                                       
                    cvs = CVS(clf.model, clf.features, clf.targets, cv = cv)
                    export_cross_val_results_clf(clf, cv, "SVC_cv", col_name, unique_categories, inputs['random state'])     

                elif data_process == 'leave one out':
                    clf.model = SVC(C=inputs['C'], kernel=inputs['kernel'], class_weight=inputs['class weight'])
      
                    export_loo_results_clf(clf, loo, "SVC_loo", col_name, unique_categories)  

        if inputs['model'] == 'BaggingClassifier':

            with col2:
                with st.expander('Operator'):
                    
                    data_process = st.selectbox('data process', ('train test split','cross val score', 'leave one out'), label_visibility='collapsed')
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif data_process == 'cross val score':
                        cv = st.number_input('cv',1,20,5)
                    elif data_process == 'leave one out':
                        loo = LeaveOneOut() 

            with st.container():
                button_train = st.button('Train', use_container_width=True)
            
            if button_train:
                if data_process == 'train test split':
             
                    clf.model = BaggingClassifier(estimator = tree.DecisionTreeClassifier(random_state=inputs['random state']),n_estimators=inputs['nestimators'],
                                                max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 

                    clf.BaggingClassifier()
                    
                    if not check_string(targets):
                        conf_matrix = confusion_matrix(clf.Ytest.values, clf.Ypred.values)
                        conf_df = pd.DataFrame(conf_matrix, index=np.unique(clf.Ytest))
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, fmt="d",cmap='Blues')
                            plt.xlabel('Actual')
                            plt.ylabel('Prediction')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)
                    else:
                        clf.Ytest[col_name[0]] = pd.Series(unique_categories).iloc[clf.Ytest[col_name[0]]].values
                        clf.Ypred[col_name[0]] = pd.Series(unique_categories).iloc[clf.Ypred[col_name[0]]].values
                        conf_matrix = confusion_matrix(clf.Ytest.values, clf.Ypred.values, labels=np.unique(clf.Ytest))
                        conf_df = pd.DataFrame(conf_matrix, index=np.unique(clf.Ytest), columns=np.unique(clf.Ytest))
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, cmap='Blues',fmt='d',
                                        xticklabels=conf_df.columns, yticklabels=conf_df.index)
                            plt.xlabel('Actual')
                            plt.ylabel('Prediction')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)
                            
                        result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'actual vs prediction.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif data_process == 'cross val score':
                    clf.model = BaggingClassifier(estimator = tree.DecisionTreeClassifier(random_state=inputs['random state']),n_estimators=inputs['nestimators'],
                                                max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
                    cvs = CVS(clf.model, clf.features, clf.targets, cv = cv)
                    export_cross_val_results_clf(clf, cv, "BaggingC_cv", col_name, unique_categories, inputs['random state']) 

                elif data_process == 'leave one out':
                    clf.model = BaggingClassifier(estimator = tree.DecisionTreeClassifier(random_state=inputs['random state']),n_estimators=inputs['nestimators'],
                                                max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 

                    export_loo_results_clf(clf, loo, "BaggingC_loo", col_name, unique_categories)  
        
        if inputs['model'] == 'AdaBoostClassifier':

            with col2:
                with st.expander('Operator'):
                    
                    data_process = st.selectbox('data process', ('train test split','cross val score', 'leave one out'), label_visibility='collapsed')
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif data_process == 'cross val score':
                        cv = st.number_input('cv',1,20,5)

                    elif data_process == 'leave one out':
                        loo = LeaveOneOut()        
            
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            
            if button_train:
                if data_process == 'train test split':
                
                    clf.model = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state'])

                    clf.AdaBoostClassifier()
                    
                    if not check_string(targets):
                        conf_matrix = confusion_matrix(clf.Ytest.values, clf.Ypred.values)
                        conf_df = pd.DataFrame(conf_matrix, index=np.unique(clf.Ytest))
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, fmt="d",cmap='Blues')
                            plt.xlabel('Actual')
                            plt.ylabel('Prediction')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)
                    else:
                        clf.Ytest[col_name[0]] = pd.Series(unique_categories).iloc[clf.Ytest[col_name[0]]].values
                        clf.Ypred[col_name[0]] = pd.Series(unique_categories).iloc[clf.Ypred[col_name[0]]].values
                        conf_matrix = confusion_matrix(clf.Ytest.values, clf.Ypred.values, labels=np.unique(clf.Ytest))
                        conf_df = pd.DataFrame(conf_matrix, index=np.unique(clf.Ytest), columns=np.unique(clf.Ytest))
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, cmap='Blues',fmt='d',
                                        xticklabels=conf_df.columns, yticklabels=conf_df.index)
                            plt.xlabel('Actual')
                            plt.ylabel('Prediction')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)

                        result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'actual vs prediction.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)
                            
                elif data_process == 'cross val score':

                    clf.model = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state'])
                    cvs = CVS(clf.model, clf.features, clf.targets, cv = cv)
                    export_cross_val_results_clf(clf, cv, "AdaBoostC_cv", col_name, unique_categories, inputs['random state']) 
                
                elif data_process == 'leave one out':
                    clf.model = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state'])
                    
                    export_loo_results_clf(clf, loo, "AdaBoostC_loo", col_name, unique_categories)  

        if inputs['model'] == 'GradientBoostingClassifier':

            with col2:
                with st.expander('Operator'):
                    
                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])
                    
                    data_process = st.selectbox('data process', ('train test split','cross val score','leave one out'), label_visibility='collapsed')
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif data_process == 'cross val score':

                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        cv = st.number_input('cv',1,20,5)
                    
                    elif data_process == 'leave one out':

                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        loo = LeaveOneOut()         
            
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            
            if button_train:
                if data_process == 'train test split':
                
                    clf.model = GradientBoostingClassifier(learning_rate=inputs['learning rate'],n_estimators=inputs['nestimators'],max_depth=inputs['max depth'],max_features=inputs['max features'],
                                                        random_state=inputs['random state'])
                    clf.GradientBoostingClassifier()
                        
                    if not check_string(targets):
                        conf_matrix = confusion_matrix(clf.Ytest.values, clf.Ypred.values)
                        conf_df = pd.DataFrame(conf_matrix, index=np.unique(clf.Ytest))
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, fmt="d",cmap='Blues')
                            plt.xlabel('Actual')
                            plt.ylabel('Prediction')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)
                    else:
                        clf.Ytest[col_name[0]] = pd.Series(unique_categories).iloc[clf.Ytest[col_name[0]]].values
                        clf.Ypred[col_name[0]] = pd.Series(unique_categories).iloc[clf.Ypred[col_name[0]]].values
                        conf_matrix = confusion_matrix(clf.Ytest.values, clf.Ypred.values, labels=np.unique(clf.Ytest))
                        conf_df = pd.DataFrame(conf_matrix, index=np.unique(clf.Ytest), columns=np.unique(clf.Ytest))
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, cmap='Blues',fmt='d',
                                        xticklabels=conf_df.columns, yticklabels=conf_df.index)
                            plt.xlabel('Actual')
                            plt.ylabel('Prediction')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)

                        result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'actual vs prediction.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif data_process == 'cross val score':
                    
                    clf.model = GradientBoostingClassifier(learning_rate=inputs['learning rate'],n_estimators=inputs['nestimators'],max_depth=inputs['max depth'],max_features=inputs['max features'],
                                                        random_state=inputs['random state'])
                    cvs = CVS(clf.model, clf.features, clf.targets, cv = cv)
                    export_cross_val_results_clf(clf, cv, "GBC_cv", col_name, unique_categories, inputs['random state'])   

                elif data_process == 'leave one out':
                    
                    clf.model = GradientBoostingClassifier(learning_rate=inputs['learning rate'],n_estimators=inputs['nestimators'],max_depth=inputs['max depth'],max_features=inputs['max features'],
                                                        random_state=inputs['random state'])
                    export_loo_results_clf(clf, loo, "GBC_loo", col_name, unique_categories)

        if inputs['model'] == 'XGBClassifier':

            with col2:
                with st.expander('Operator'):
                                   
                    data_process = st.selectbox('data process', ('train test split','cross val score', 'leave one out'), label_visibility='collapsed')
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  

                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif data_process == 'cross val score':
                        cv = st.number_input('cv',1,20,5)
                    elif data_process == 'leave one out':
                        loo = LeaveOneOut() 
            
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            
            if button_train:
                if data_process == 'train test split':
                
                    clf.model = xgb.XGBClassifier(booster=inputs['base estimator'], n_estimators=inputs['nestimators'], 
                                                subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                learning_rate=inputs['learning rate'])
                    clf.Ytest = clf.Ytest.reset_index(drop=True)
                
                    clf.XGBClassifier()
                        
                    if not check_string(targets):
                        conf_matrix = confusion_matrix(clf.Ytest.values, clf.Ypred.values)
                        conf_df = pd.DataFrame(conf_matrix, index=np.unique(clf.Ytest))
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, fmt="d",cmap='Blues')
                            plt.xlabel('Actual')
                            plt.ylabel('Prediction')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)
                    else:
                        clf.Ytest[col_name[0]] = pd.Series(unique_categories).iloc[clf.Ytest[col_name[0]]].values
                        clf.Ypred[col_name[0]] = pd.Series(unique_categories).iloc[clf.Ypred[col_name[0]]].values
                        conf_matrix = confusion_matrix(clf.Ytest.values, clf.Ypred.values, labels=np.unique(clf.Ytest))
                        conf_df = pd.DataFrame(conf_matrix, index=np.unique(clf.Ytest), columns=np.unique(clf.Ytest))
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, cmap='Blues',fmt='d',
                                        xticklabels=conf_df.columns, yticklabels=conf_df.index)
                            plt.xlabel('Actual')
                            plt.ylabel('Prediction')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)

                        result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'actual vs prediction.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif data_process == 'cross val score':
                        
                        clf.model = xgb.XGBClassifier(booster=inputs['base estimator'], n_estimators=inputs['nestimators'], 
                                                    subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                    learning_rate=inputs['learning rate'])
                        
                        cvs = CVS(clf.model, clf.features, clf.targets, cv = cv)
                        export_cross_val_results_clf(clf, cv, "XGBC_cv", col_name, unique_categories, inputs['random state'])  
                
                elif data_process == 'leave one out':
                        
                        clf.model = xgb.XGBClassifier(booster=inputs['base estimator'], n_estimators=inputs['nestimators'], 
                                                    subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                    learning_rate=inputs['learning rate'])

                        export_loo_results_clf(clf, loo, "XGBC_loo", col_name, unique_categories)
        
        if inputs['model'] == 'CatBoostClassifier':

            with col2:
                with st.expander('Operator'):
                    
                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])
                    
                    data_process = st.selectbox('data process', ('train test split','cross val score','leave one out'), label_visibility='collapsed')
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif data_process == 'cross val score':

                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        cv = st.number_input('cv',1,20,5)

                    elif data_process == 'leave one out':

                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        loo = LeaveOneOut() 

            with st.container():
                button_train = st.button('Train', use_container_width=True)
            
            if button_train:
                if data_process == 'train test split':
                
                    clf.model = CatBoostClassifier(iterations=inputs['niteration'],learning_rate=inputs['learning rate'],depth = inputs['max depth'])

                    clf.CatBoostClassifier()
                        
                    if not check_string(targets):
                        conf_matrix = confusion_matrix(clf.Ytest.values, clf.Ypred.values)
                        conf_df = pd.DataFrame(conf_matrix, index=np.unique(clf.Ytest))
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, fmt="d",cmap='Blues')
                            plt.xlabel('Actual')
                            plt.ylabel('Prediction')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)
                    else:
                        clf.Ytest[col_name[0]] = pd.Series(unique_categories).iloc[clf.Ytest[col_name[0]]].values
                        clf.Ypred[col_name[0]] = pd.Series(unique_categories).iloc[clf.Ypred[col_name[0]]].values
                        conf_matrix = confusion_matrix(clf.Ytest.values, clf.Ypred.values, labels=np.unique(clf.Ytest))
                        conf_df = pd.DataFrame(conf_matrix, index=np.unique(clf.Ytest), columns=np.unique(clf.Ytest))
                        with plt.style.context(['nature','no-latex']):
                            fig, ax = plt.subplots()
                            sns.heatmap(conf_df, annot=True, cmap='Blues',fmt="d",
                                        xticklabels=conf_df.columns, yticklabels=conf_df.index)
                            plt.xlabel('Actual')
                            plt.ylabel('Prediction')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)

                        result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'actual vs prediction.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif data_process == 'cross val score':
                        
                        clf.model = CatBoostClassifier(iterations=inputs['niteration'],learning_rate=inputs['learning rate'],depth = inputs['max depth'])

                        cvs = CVS(clf.model, clf.features, clf.targets, cv = cv)

                        export_cross_val_results_clf(clf, cv, "CatBoostC_cv", col_name, unique_categories, inputs['random state'])  
                
                elif data_process == 'leave one out':
                        
                        clf.model = CatBoostClassifier(iterations=inputs['niteration'],learning_rate=inputs['learning rate'],depth = inputs['max depth'])

                        export_loo_results_clf(clf, loo, "CatBoostC_loo", col_name, unique_categories)

        st.write('---')

elif select_option == "聚类降维":
    colored_header(label="聚类降维",description=" ",color_name="violet-90")
    file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
    if file is None:
        table = PrettyTable(['上传文件名称', '名称','数据说明'])
        table.add_row(['file_1','dataset','数据集'])
        st.write(table)
    if file is not None:
        df = pd.read_csv(file)
        check_string_NaN(df)
        colored_header(label="数据信息", description=" ",color_name="violet-70")
        nrow = st.slider("rows", 1, len(df), 5)
        df_nrow = df.head(nrow)
        st.write(df_nrow)

        colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

        target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)
        
        col_feature, col_target = st.columns(2)
            
        # features
        features = df.iloc[:,:-target_num]
        # targets
        targets = df.iloc[:,-target_num:]
        with col_feature:    
            st.write(features.head())
        with col_target:   
            st.write(targets.head())
        
       #=============== cluster ================

        colored_header(label="Cluster",description=" ",color_name="violet-70")
        cluster = CLUSTER(features, targets)

        # colored_header(label="Choose Target", description=" ", color_name="violet-30")
        # target_selected_option = st.selectbox('target', list(cluster.targets)[::-1])

        # cluster.targets = targets[target_selected_option]

        model_path = './models/cluster'

        colored_header(label="Training", description=" ",color_name="violet-30")

        template_alg = model_platform(model_path)

        inputs, col2 = template_alg.show()
    
        if inputs['model'] == 'K-means':

            with col2:
                pass

            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                
                    cluster.model = KMeans(n_clusters=inputs['n clusters'], random_state = inputs['random state'])
                    
                    cluster.K_means()

                    clustered_df = pd.concat([cluster.features,pd.DataFrame(cluster.model.labels_)], axis=1)
                    
                    r_name='cluster label'
                    c_name=clustered_df.columns[-1]
                    
                    clustered_df.rename(columns={c_name:r_name},inplace=True)
                    with st.expander('聚类结果'):
                        st.write(clustered_df)
        if inputs['model'] == 'PCA':   
            with col2:
                pass
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:  
                pca_all = PCA(n_components=cluster.features.shape[1])
                pca_all.fit(cluster.features)
                with plt.style.context(['nature','no-latex']):
                    fig, ax = plt.subplots()
                    ax = plt.plot(np.cumsum(pca_all.explained_variance_ratio_ * 100))
                    plt.grid()
                    plt.xlabel('Numbers of components')
                    plt.ylabel('Explained variance')
                    st.pyplot(fig)
                def std_PCA(**argv):
                    scaler = MinMaxScaler()
                    pca = PCA(**argv)
                    pipeline = Pipeline([('scaler',scaler),('pca',pca)])
                    return pipeline
                
                PCA_model = std_PCA(n_components=inputs['ncomponents'])
                PCA_transformed_data = PCA_model.fit_transform(cluster.features)
                if inputs['ncomponents'] == 2:
                    with plt.style.context(['nature','no-latex']):
                        fig, ax = plt.subplots()
                        ax = plt.scatter(PCA_transformed_data[:,0], PCA_transformed_data[:,1], c=[int(i) for i in cluster.targets.values], s=2, cmap='tab10')
                        plt.xlabel('1st dimension')
                        plt.ylabel('2st dimension')
                        plt.tight_layout()
                        st.pyplot(fig)   
                    result_data =  PCA_transformed_data
                    with st.expander('降维结果'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'dim reduction data.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)                    
                else:
                    result_data =  PCA_transformed_data
                    with st.expander('降维结果'):
                        st.write(result_data)                    
                        tmp_download_link = download_button(result_data, f'dim reduction data.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)   
        if inputs['model'] == 'TSEN':   
            with col2:
                pass
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:  
                TSNE_model = TSNE(n_components=inputs['ncomponents'], perplexity=inputs['perplexity'], learning_rate='auto',n_iter=inputs['max iter'], init='pca', random_state=inputs['random state'])
                TSNE_transformed_data  = TSNE_model.fit_transform(cluster.features)
                
                if inputs['ncomponents'] == 2:
                    with plt.style.context(['nature','no-latex']):
                        fig, ax = plt.subplots()
                        ax = plt.scatter(TSNE_transformed_data[:,0], TSNE_transformed_data[:,1], c=[int(i) for i in cluster.targets.values], s=2, cmap='tab10')
                        plt.xlabel('1st dimension')
                        plt.ylabel('2st dimension')
                        plt.tight_layout()
                        st.pyplot(fig)   
                    result_data =  TSNE_transformed_data
                    with st.expander('降维结果'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'dim reduction data.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)                    
                else:
                    result_data =  TSNE_transformed_data
                    with st.expander('降维结果'):
                        st.write(result_data)                    
                        tmp_download_link = download_button(result_data, f'dim reduction data.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)           
        st.write('---')   

elif select_option == "主动学习":
    with st.sidebar:
        sub_option = option_menu(None, ["单目标主动学习", "多目标主动学习"])
    
    if sub_option == "单目标主动学习":

        colored_header(label="单目标主动学习",description=" ",color_name="violet-90")

        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed", accept_multiple_files=True)
        if len(file) != 2:
            table = PrettyTable(['上传文件名称', '名称','数据说明'])
            table.add_row(['file_1','dataset','数据集'])
            table.add_row(['file_2','visual data','虚拟采样点'])
            st.write(table)      
        if len(file) == 2:
            
            colored_header(label="数据信息",description=" ",color_name="violet-70")

            # with st.expander('Data Information'):
            df = pd.read_csv(file[0])
            # if len(file) == 2:
            df_vs = pd.read_csv(file[1])
            check_string_NaN(df)

            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

            target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)
            
            col_feature, col_target = st.columns(2)
            
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())

            sp = SAMPLING(features, targets)

            colored_header(label="选择目标变量", description=" ", color_name="violet-70")

            target_selected_option = st.selectbox('target', list(sp.targets))
            
            sp.targets = sp.targets[target_selected_option]
            
            colored_header(label="Sampling", description=" ",color_name="violet-70")

            model_path = './models/active learning'

            template_alg = model_platform(model_path)

            inputs, col2 = template_alg.show()

            if inputs['model'] == 'BayeSampling':

                with col2:
                    # if len(file) == 2:
                    sp.vsfeatures = df_vs
                    st.info('You have upoaded the visual sample point file.')
                    feature_name = sp.features.columns.tolist()
                
                with st.expander('visual samples'):
                    st.write(sp.vsfeatures)
                    tmp_download_link = download_button(sp.vsfeatures, f'visual samples.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                Bgolearn = BGOS.Bgolearn()
                
                colored_header(label="Optimize", description=" ",color_name="violet-70")
                with st.container():
                    button_train = st.button('Train', use_container_width=True)
                if button_train:
                    Mymodel = Bgolearn.fit(data_matrix = sp.features, Measured_response = sp.targets, virtual_samples = sp.vsfeatures,
                                        opt_num=inputs['opt num'], min_search=inputs['min search'], noise_std= float(inputs['noise std']))
                    if inputs['sample criterion'] == 'Expected Improvement algorith':
                        res = Mymodel.EI()
                        
                    elif inputs['sample criterion'] == 'Expected improvement with "plugin"':
                        res = Mymodel.EI_plugin()

                    elif inputs['sample criterion'] == 'Augmented Expected Improvement':
                        with st.expander('EI HyperParamters'):
                            alpha = st.slider('alpha', 0.0, 3.0, 1.0)
                            tao = st.slider('tao',0.0, 1.0, 0.0)
                        res = Mymodel.Augmented_EI(alpha = alpha, tao = tao)

                    elif inputs['sample criterion'] == 'Expected Quantile Improvement':
                        with st.expander('EQI HyperParamters'):
                            beta= st.slider('beta',0.2, 0.8, 0.5)
                            tao = st.slider('tao_new',0.0, 1.0, 0.0)            
                        res = Mymodel.EQI(beta = beta,tao_new = tao)

                    elif inputs['sample criterion'] == 'Reinterpolation Expected Improvement':  
                        res = Mymodel.Reinterpolation_EI() 

                    elif inputs['sample criterion'] == 'Upper confidence bound':
                        with st.expander('UCB HyperParamters'):
                            alpha = st.slider('alpha', 0.0, 3.0, 1.0)
                        res = Mymodel.UCB(alpha=alpha)

                    elif inputs['sample criterion'] == 'Probability of Improvement':
                        with st.expander('PoI HyperParamters'):
                            tao = st.slider('tao',0.0, 0.3, 0.0)
                        res = Mymodel.PoI(tao = tao)

                    elif inputs['sample criterion'] == 'Predictive Entropy Search':
                        with st.expander('PES HyperParamters'):
                            sam_num = st.number_input('sample number',100, 1000, 500)
                        res = Mymodel.PES(sam_num = sam_num)  
                        
                    elif inputs['sample criterion'] == 'Knowledge Gradient':
                        with st.expander('Knowldge_G Hyperparameters'):
                            MC_num = st.number_input('MC number', 50,300,50)
                        res = Mymodel.Knowledge_G(MC_num = MC_num) 

                    elif inputs['sample criterion'] == 'Least Confidence':
                        
                        Mymodel = Bgolearn.fit(Mission='Classification', Classifier=inputs['Classifier'], data_matrix = sp.features, Measured_response = sp.targets, virtual_samples = sp.vsfeatures,
                                        opt_num=inputs['opt num'])
                        res = Mymodel.Least_cfd() 
        
                    elif inputs['sample criterion'] == 'Margin Sampling':
                        Mymodel = Bgolearn.fit(Mission='Classification', Classifier=inputs['Classifier'], data_matrix = sp.features, Measured_response = sp.targets, virtual_samples = sp.vsfeatures,
                                opt_num=inputs['opt num'])
                        res = Mymodel.Margin_S()

                    elif inputs['sample criterion'] == 'Entropy-based approach':
                        Mymodel = Bgolearn.fit(Mission='Classification', Classifier=inputs['Classifier'], data_matrix = sp.features, Measured_response = sp.targets, virtual_samples = sp.vsfeatures,
                                opt_num=inputs['opt num'])
                        res = Mymodel.Entropy()

                    st.info('Recommmended Sample')
                    sp.sample_point = pd.DataFrame(res[1], columns=feature_name)
                    st.write(sp.sample_point)
                    tmp_download_link = download_button(sp.sample_point, f'recommended samples.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)

    elif sub_option == "多目标主动学习":

        colored_header(label="多目标主动学习",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed", accept_multiple_files=True)
        if len(file) != 2:
            table = PrettyTable(['上传文件名称', '名称','数据说明'])
            table.add_row(['file_1','dataset','数据集'])
            table.add_row(['file_2','visual data','虚拟采样点'])
            st.write(table)
        elif len(file) == 2:    
            colored_header(label="数据信息",description=" ",color_name="violet-70")
            # with st.expander('Data Information'):
            df = pd.read_csv(file[0])
        
            df_vs = pd.read_csv(file[1])
            check_string_NaN(df_vs)
            check_string_NaN(df)

            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)
            colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

            target_num = st.number_input('目标变量数量',  min_value=2, max_value=10, value=2)
            
            col_feature, col_target = st.columns(2)
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())
            
            col_feature, col_target = st.columns(2)

    # =================== model ====================================
            reg = REGRESSOR(features,targets)

            colored_header(label="选择目标变量", description=" ", color_name="violet-70")
            target_selected_option = st.multiselect('target', list(reg.targets)[::-1], default=targets.columns.tolist())
            
            reg.targets = targets[target_selected_option]
            reg.Xtrain = features
            reg.Ytrain = targets
            feature_name = reg.Xtrain.columns
            colored_header(label="Sampling", description=" ",color_name="violet-30")
            model_path = './models/multi-obj'

            template_alg = model_platform(model_path)
            inputs, col2 = template_alg.show()  

            if inputs['model'] == 'MOBO':

                mobo = Mobo4mat()

                with col2:
                    # if len(file) == 2:
                    # features
                    # vs_features = df_vs.iloc[:,:-target_num]
                    vs_features = df_vs
                    # targets
                    # vs_targets = df_vs.iloc[:,-target_num:]
                    reg.Xtest = vs_features
                    st.info('You have upoaded the visual sample point file.')
                    # else:
                    #     feature_name = features.columns.tolist()
                    #     mm = MinMaxScaler()
                    #     mm.fit(features)
                    #     data_min = mm.data_min_
                    #     data_max = mm.data_max_
                    #     trans_features = mm.transform(features)
                    #     min_ratio, max_ratio = st.slider('sample space ratio', 0.8, 1.2, (1.0, 1.0))
            
                    #     sample_num = st.selectbox('sample number', ['10','20','50','80','100'])
                    #     feature_num = trans_features.shape[1]

                    #     vs = np.linspace(min_ratio * data_min, max_ratio *data_max, int(sample_num))  
                    #     reg.Xtest = pd.DataFrame(vs, columns=feature_name)


                    if inputs['normalize'] == 'StandardScaler':
                        reg.X = pd.concat([reg.Xtrain, reg.Xtest])
                        reg.X, scaler = normalize(reg.X, "StandardScaler")
                        reg.X = pd.DataFrame(reg.X, columns=feature_name)  
                        reg.Xtrain = reg.X.iloc[:len(reg.Xtrain),:]     
                        reg.Xtest = reg.X.iloc[len(reg.Xtrain):,:].reset_index(drop=True)
                    elif inputs['normalize'] == 'MinMaxScaler':
                        reg.X = pd.concat([reg.Xtrain, reg.Xtest])
                        reg.X, scaler = normalize(reg.X, "StandardScaler")
                        reg.X = pd.DataFrame(reg.X, columns=feature_name)  
                        reg.Xtrain = reg.X.iloc[:len(reg.Xtrain),:]     
                        reg.Xtest = reg.X.iloc[len(reg.Xtrain):,:].reset_index(drop=True) 

                pareto_front = find_non_dominated_solutions(reg.targets.values, target_selected_option)
                pareto_front = pd.DataFrame(pareto_front, columns=target_selected_option)

                if inputs['objective'] == 'max':  
                    # st.write(type(reg.targets.values))  
                    reg.targets = - reg.targets
                    pareto_front = find_non_dominated_solutions(reg.targets.values, target_selected_option)
                    pareto_front = pd.DataFrame(pareto_front, columns=target_selected_option)
                    pareto_front = -pareto_front
                    reg.targets = -reg.targets
    
                col1, col2 = st.columns([2, 1])
                with col1:
                    with plt.style.context(['nature','no-latex']):
                        fig, ax = plt.subplots()
                        ax.plot(pareto_front[target_selected_option[0]], pareto_front[target_selected_option[1]], 'k--')
                        ax.scatter(reg.targets[target_selected_option[0]], reg.targets[target_selected_option[1]])
                        ax.set_xlabel(target_selected_option[0])
                        ax.set_ylabel(target_selected_option[1])
                        ax.set_title('Pareto front of visual space')
                        st.pyplot(fig)
                with col2:
                    st.write(pareto_front)
                    tmp_download_link = download_button(pareto_front, f'Pareto_front.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                
                ref_point = []
                for i in range(len(target_selected_option)):
                    ref_point_loc = st.number_input(target_selected_option[i] + ' ref location', 0, 100000, 50)
                    ref_point.append(ref_point_loc)
                colored_header(label="Optimize", description=" ",color_name="violet-70")
                with st.container():
                    button_train = st.button('Opt', use_container_width=True)  
                if button_train:      
                    if reg.Xtrain.columns.tolist() != reg.Xtest.columns.tolist():
                        st.error('the feature number in Visual sample file is wrong')
                        st.stop()
                    HV_value, recommend_point = mobo.fit(X = reg.Xtrain, y = reg.Ytrain, visual_data=reg.Xtest, 
                                                    method=inputs['method'],number= inputs['num'], objective=inputs['objective'], ref_point=ref_point)
                    HV_value = pd.DataFrame(HV_value, columns=["HV value"]) 
                    st.write(HV_value)
                    recommend_point = pd.DataFrame(recommend_point, columns=feature_name)  
                    
                    if inputs['normalize'] == 'StandardScaler':
                        recommend_point  = inverse_normalize(recommend_point, scaler, "StandardScaler")
                    elif inputs['normalize'] == 'MinMaxScaler':
                        recommend_point  = inverse_normalize(recommend_point, scaler, "MinMaxScaler")
                    
                    st.write(recommend_point)
                    tmp_download_link = download_button(recommend_point, f'推荐试验样本.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True) 
                
                with st.expander('visual samples'):
                    if inputs['normalize'] == 'StandardScaler':
                        reg.Xtest  = inverse_normalize(reg.Xtest, scaler, "StandardScaler")
                    elif inputs['normalize'] == 'MinMaxScaler':
                        reg.Xtest  = inverse_normalize(reg.Xtest, scaler, "MinMaxScaler")
                    st.write(reg.Xtest)
                    tmp_download_link = download_button(reg.Xtest, f'visual samples.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)                   

elif select_option == "迁移学习":
    with st.sidebar:
        # sub_option = option_menu(None, ["Boosting", "Neural Network"])
        sub_option = option_menu(None, ["Boosting"])
    if sub_option == "Boosting":
        colored_header(label="基于boosting迁移学习",description=" ",color_name="violet-90")
        # sub_sub_option = option_menu(None, ["样本迁移","特征迁移","参数迁移"],
        #                         icons=['list-task',  "list-task","list-task"],
        #                         menu_icon="cast", default_index=0, orientation="horizontal")
        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed", accept_multiple_files=True)
        if len(file) < 3:
            table = PrettyTable(['上传文件名称', '名称','数据说明'])
            table.add_row(['file_1','test_data','无标签目标域数据'])
            table.add_row(['file_2','target_data','有标签目标域数据'])
            table.add_row(['file_3','source_data_1','1 源域数据'])
            table.add_row(['...','...','...'])
            table.add_row(['file_n','source_data_n','n 源域数据'])
            st.write(table)
        # elif len(file) == 3:
        #     df_test = pd.read_csv(file[0])
        #     df_target = pd.read_csv(file[1])
        #     df_source = pd.read_csv(file[2])
        elif len(file) >= 3:
            df_test = pd.read_csv(file[0])
            df_target = pd.read_csv(file[1])
            source_files = file[2:]
            df = [pd.read_csv(f) for f in source_files]
            df_source = pd.concat(df, axis=0)

            colored_header(label="数据信息", description=" ",color_name="violet-70")

            # show target data
            nrow = st.slider("rows", 1, len(df_target), 5)
            df_nrow = df_target.head(nrow)
            st.write(df_nrow)
            colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

            target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)
            
            col_feature, col_target = st.columns(2)
            # features
            target_features = df_target.iloc[:,:-target_num]
            source_features = df_source.iloc[:,:-target_num]            
            # targets
            target_targets = df_target.iloc[:,-target_num:]
            source_targets = df_source.iloc[:,-target_num:]

            with col_feature:    
                st.write(target_features.head())
            with col_target:   
                st.write(target_targets.head())
    # =================== model ====================================
            # features = pd.concat([source_features, target_features], axis=0)
            # targets = pd.concat([source_targets,target_targets], axis=0)
            reg = REGRESSOR(target_features, target_targets)

            colored_header(label="选择目标变量", description=" ", color_name="violet-70")

            target_selected_option = st.selectbox('target', list(reg.targets)[::-1])

            reg.targets = target_targets[target_selected_option]

            colored_header(label="Transfer", description=" ",color_name="violet-30")

            model_path = './models/transfer learning'

            template_alg = model_platform(model_path)

            inputs, col2 = template_alg.show()
            # TrAdaBoostR2 = template_alg.TrAdaBoostR2(reg.)
            with col2:
                if inputs['max iter'] > source_features.shape[0]:
                    st.warning('The maximum of iterations should be smaller than %d' % source_features.shape[0])

            if inputs['model'] == 'TrAdaboostR2':
                TrAdaboostR2 = TrAdaboostR2()
                with st.container():
                    button_train = st.button('Train', use_container_width=True)
                if button_train:
                    TrAdaboostR2.fit(inputs, source_features, target_features, source_targets[target_selected_option], target_targets[target_selected_option], inputs['max iter'])
                    
                    Xtest = df_test[list(target_features.columns)]
                    predict = TrAdaboostR2.predict(Xtest)
                    prediction = pd.DataFrame(predict, columns=[reg.targets.name])
                    try:
                        Ytest = df_test[target_selected_option]
                        plot = customPlot()
                        plot.pred_vs_actual(Ytest, prediction)
                        r2 = r2_score(Ytest, prediction)
                        st.write('R2: {}'.format(r2))
                        result_data = pd.concat([Ytest, pd.DataFrame(prediction)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('预测结果'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'预测结果.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)
                    except KeyError:
                        st.write(prediction)
                        tmp_download_link = download_button(prediction, f'预测结果.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)         
                    with st.expander("基学习器及权重下载"):
                        # with open('estimator_wetesrt.pickle', 'wb') as file:
                        #     pickle.dump(TrAdaboostR2.estimator_weight, file)
                        st.write(TrAdaboostR2.estimator_weight)
                        model_name = 'estimators'
                        tmp_download_link = download_button(TrAdaboostR2.estimators, model_name+f'.pickle', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)
                        data_name = 'estimator_weights'       
                        tmp_download_link = download_button(TrAdaboostR2.estimator_weight, data_name+f'.pickle', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)        

            elif inputs['model'] == 'TwoStageTrAdaboostR2':
                st.write('Please wait...')
            elif inputs['model'] == 'TwoStageTrAdaboostR2-revised':
                st.write('Please wait...')
                st.write('---')

    # elif sub_option == "Neural Network":
    #     colored_header(label="基于Neural Network迁移学习",description=" ",color_name="violet-90")
    #     sub_sub_option = option_menu(None, ["样本迁移","特征迁移","参数迁移"],
    #                             icons=['list-task',  "list-task","list-task"],
    #                             menu_icon="cast", default_index=0, orientation="horizontal")      

elif select_option == "代理优化":
    with st.sidebar:
        sub_option = option_menu(None, ["单目标代理优化", "多目标代理优化","迁移学习-单目标代理优化","迁移学习-多目标代理优化"])
    if sub_option == "单目标代理优化":

        colored_header(label="单目标代理优化（成分/工艺）",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.pickle` model and `.csv` file",  label_visibility="collapsed", accept_multiple_files=True)
        if len(file) < 3:
            table = PrettyTable(['上传文件名称', '名称','数据说明'])
            table.add_row(['file_1','dataset','数据集'])
            table.add_row(['file_2','boundary','设计变量上下界'])
            table.add_row(['file_3','model','模型'])
            st.write(table)
        
        if len(file) >= 3:
            df = pd.read_csv(file[0])
            check_string_NaN(df)

            colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

            target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)
            
            col_feature, col_target = st.columns(2)
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())

            df_var = pd.read_csv(file[1])
            features_name = df_var.columns.tolist()
            range_var = df_var.values
            vars_min = get_column_min(range_var)
            vars_max = get_column_max(range_var)
            array_vars_min = np.array(vars_min).reshape(1,-1)
            array_vars_max = np.array(vars_max).reshape(1,-1)
            vars_bound = np.concatenate([array_vars_min, array_vars_max], axis=0)
            colored_header(label="特征变量范围", description=" ", color_name="violet-70")
            vars_bound = pd.DataFrame(vars_bound, columns=features_name)
            st.write(vars_bound)
        
            colored_header(label="Optimize", description=" ", color_name="violet-70")
            model = pickle.load(file[2])
            model_path = './models/surrogate optimize'
            template_alg = model_platform(model_path)

            inputs, col2 = template_alg.show()
            inputs['lb'] = vars_min
            inputs['ub'] = vars_max
            with col2:
                preprocess = st.selectbox('data preprocess',[None, 'StandardScaler','MinMaxScaler'])
                data = pd.concat([features,vars_bound])
                # st.write(data)
                if preprocess == 'StandardScaler':
                    features, scaler = normalize(data, 'StandardScaler')
                    vars_bound = features.tail(2)
                elif preprocess == 'MinMaxScaler':
                    features, scaler = normalize(data, 'MinMaxScaler')
                    vars_bound = features.tail(2)

            if not len(inputs['lb']) == len(inputs['ub']) == inputs['n dim']:
                st.warning('the variable number should be %d' % vars_bound.shape[1])
            else:
                st.info("the variable number is correct")

            with st.container():
                button_train = st.button('Opt', use_container_width=True)
            if button_train:  
                def opt_func(x):
                    x = x.reshape(1,-1)
                    y_pred = model.predict(x)
                    if inputs['objective'] == 'max':
                        y_pred = -y_pred
                    return y_pred
                
                plot = customPlot()  
                if inputs['model'] == 'PSO':    
                    
                    alg = PSO(func=opt_func, dim=inputs['n dim'], pop=inputs['size pop'], max_iter=inputs['max iter'], lb=inputs['lb'], ub=inputs['ub'],
                            w=inputs['w'], c1=inputs['c1'], c2=inputs['c2'])
            
                    alg.run()
                    best_x = alg.gbest_x
                    best_y = alg.gbest_y

                    loss_history = alg.gbest_y_hist
                    if inputs['objective'] == 'max':
                        loss_history = -np.array(loss_history)                    

                    st.info('Recommmended Sample')
                    truncate_func = np.vectorize(lambda x: '{:,.4f}'.format(x))
                    best_x = truncate_func(best_x).reshape(1,-1)

                    best_x = pd.DataFrame(best_x, columns = features_name)
                    if preprocess == 'StandardScaler':
                        best_x = inverse_normalize(best_x, scaler, 'StandardScaler')
                    elif preprocess == 'MinMaxScaler':
                        best_x = inverse_normalize(best_x, scaler, 'MinMaxScaler')

                    st.write(best_x)         
                    tmp_download_link = download_button(best_x, f'recommended samples.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                    if inputs['objective'] == 'max':
                        best_= -best_y
                    st.info('PSO best_y: %s' % best_y.item())
                    plot.evolutionary_history(loss_history, 'PSO')
                    loss_history = pd.DataFrame(loss_history)
                    tmp_download_link = download_button(loss_history, f'evolutionary history.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                elif inputs['model'] == 'GA':

                    alg = GA(pop_size=inputs['size pop'], 
                            crossover=SBX(prob=0.9, eta=15),
                            mutation=PM(eta=20),
                            eliminate_duplicates=True)

                    termination = get_termination("n_gen", inputs['max iter'])                    
                    class MyProblem(ElementwiseProblem):
                        def __init__(self):
                            super().__init__(n_var=inputs['n dim'],
                                            n_obj=1,
                                            xl=np.array(inputs['lb']),
                                            xu=np.array(inputs['ub']))
                        def _evaluate(self, x, out, *args, **kwargs):
                            x = x.reshape(1,-1)
                            y_pred = model.predict(x)
                            if inputs['objective'] == 'max':
                                y_pred = -y_pred
                            out["F"] = y_pred
                            
                    problem = MyProblem()                    
                    res = minimize(problem,
                                    alg,
                                    termination,
                                    seed=1,
                                    save_history=True,
                                    verbose=False)
                    if inputs['objective'] == 'max':
                        best_y = -res.F
                    else:
                        best_y = res.F
                    best_x = res.X
                    hist = res.history
                    hist_F = []              # the objective space values in each generation
                    # st.write("Best solution found: \nX = %s\nF = %s" % (best_x, y_pred))
                    for algo in hist:
                        # retrieve the optimum from the algorithm
                        opt = algo.opt
                        # filter out only the feasible and append and objective space values
                        feas = np.where(opt.get("feasible"))[0]
                        hist_F.append(opt.get("F")[feas])
                    # replace this line by `hist_cv` if you like to analyze the least feasible optimal solution and not the population
                    if inputs['objective'] == 'max':
                        loss_history = - np.array(hist_F).reshape(-1,1)
                    else:
                        loss_history = np.array(hist_F).reshape(-1,1)
                    st.info('Recommmended Sample')
                    truncate_func = np.vectorize(lambda x: '{:,.4f}'.format(x))
                    best_x = truncate_func(best_x).reshape(1,-1)
                    
                    best_x = pd.DataFrame(best_x, columns = features_name)
                    if preprocess == 'StandardScaler':
                        best_x = inverse_normalize(best_x, scaler, 'StandardScaler')
                    elif preprocess == 'MinMaxScaler':
                        best_x = inverse_normalize(best_x, scaler, 'MinMaxScaler')

                    st.write(best_x)         
                    tmp_download_link = download_button(best_x, f'recommended samples.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                    st.info('GA best_y: %s' %  best_y.item())
                    plot.evolutionary_history(loss_history, 'GA')   
                    loss_history = pd.DataFrame(loss_history)
                    tmp_download_link = download_button(loss_history, f'evolutionary history.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                elif inputs['model'] == 'DE':
                    alg = DE(func=opt_func, n_dim=inputs['n dim'], size_pop=inputs['size pop'], max_iter=inputs['max iter'], lb=inputs['lb'], ub=inputs['ub'],
                            prob_mut = inputs['prob mut'], F=inputs['F'])

                    best_x, best_y = alg.run()

                    loss_history = alg.generation_best_Y
                    if inputs['objective'] == 'max':
                        loss_history = -np.array(loss_history)     

                    st.info('Recommmended Sample')
                    truncate_func = np.vectorize(lambda x: '{:,.4f}'.format(x))
                    best_x = truncate_func(best_x).reshape(1,-1)
                    

                    best_x = pd.DataFrame(best_x, columns = features_name)

                    if preprocess == 'StandardScaler':
                        best_x = inverse_normalize(best_x, scaler, 'StandardScaler')
                    elif preprocess == 'MinMaxScaler':
                        best_x = inverse_normalize(best_x, scaler, 'MinMaxScaler')

                    st.write(best_x)         
                    tmp_download_link = download_button(best_x, f'recommended samples.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                    if inputs['objective'] == 'max':
                        best_y = -best_y               
                    st.info('DE best_y: %s' %  best_y.item())    
                    plot.evolutionary_history(loss_history, 'DE')  
                    loss_history = pd.DataFrame(loss_history)
                    tmp_download_link = download_button(loss_history, f'evolutionary history.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                elif inputs['model'] == 'SA':
                    x0 = calculate_mean(inputs['lb'], inputs['ub'])
                    alg = SAFast(func=opt_func, x0=x0, T_max = inputs['T max'], q=inputs['q'], L=inputs['L'], max_stay_counter=inputs['max stay counter'],
                                lb=inputs['lb'], ub=inputs['ub'])

                    best_x, best_y = alg.run()

                    loss_history = alg.generation_best_Y
                    if inputs['objective'] == 'max':
                        loss_history = -np.array(loss_history)     

                    st.info('Recommmended Sample')
                    truncate_func = np.vectorize(lambda x: '{:,.4f}'.format(x))
                    best_x = truncate_func(best_x).reshape(1,-1)

                    best_x = pd.DataFrame(best_x, columns = features_name)
                    if preprocess == 'StandardScaler':
                        best_x = inverse_normalize(best_x, scaler, 'StandardScaler')
                    elif preprocess == 'MinMaxScaler':
                        best_x = inverse_normalize(best_x, scaler, 'MinMaxScaler')
                    
                    st.write(best_x)         
                    tmp_download_link = download_button(best_x, f'recommended samples.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                    if inputs['objective'] == 'max':
                        best_y = -best_y  
                    st.info('SA best_y: %s' %  best_y.item()) 
                    plot.evolutionary_history(loss_history, 'SA')  

                    loss_history = pd.DataFrame(loss_history)
                    tmp_download_link = download_button(loss_history, f'evolutionary history.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                
    elif sub_option == "多目标代理优化":

        colored_header(label="多目标代理优化（成分/工艺）",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.pickle` model and `.csv` file",  label_visibility="collapsed", accept_multiple_files=True)
        if len(file) < 4:
            table = PrettyTable(['上传文件名称', '名称','数据说明'])
            table.add_row(['file_1','dataset','数据集'])
            table.add_row(['file_2','boundary','设计变量上下界'])
            table.add_row(['file_3','model_1','目标1 模型'])
            table.add_row(['file_4','model_2','目标2 模型'])
            table.add_row(['file_5','...','...'])
            st.write(table)
        elif len(file) >= 4:        
            df = pd.read_csv(file[0])
            check_string_NaN(df)
            colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

            target_num = st.number_input('目标变量数量',  min_value=2, max_value=10, value=2)
            
            col_feature, col_target = st.columns(2)
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())
            colored_header(label="选择目标变量", description=" ", color_name="violet-70")
            target_selected_option = st.multiselect('target', list(targets)[::-1], default=targets.columns.tolist())
            df_var = pd.read_csv(file[1])
            features_name = df_var.columns.tolist()
            range_var = df_var.values
            vars_min = get_column_min(range_var)
            vars_max = get_column_max(range_var)
            array_vars_min = np.array(vars_min).reshape(1,-1)
            array_vars_max = np.array(vars_max).reshape(1,-1)
            vars_bound = np.concatenate([array_vars_min, array_vars_max], axis=0)
            colored_header(label="特征变量范围", description=" ", color_name="violet-70")
            vars_bound = pd.DataFrame(vars_bound, columns=features_name)
            st.write(vars_bound)

            colored_header(label="Optimize", description=" ", color_name="violet-70")
            model_1 = pickle.load(file[2])
            model_2 = pickle.load(file[3])
            model_path = './models/moo'
            template_alg = model_platform(model_path)

            inputs, col2 = template_alg.show()
            inputs['lb'] = vars_min
            inputs['ub'] = vars_max
    
            with col2:
                preprocess = st.selectbox('data preprocess',[None, 'StandardScaler','MinMaxScaler'])
                data = pd.concat([features,vars_bound])
                if preprocess == 'StandardScaler':
                    features, scaler = normalize(data, 'StandardScaler')
                    vars_bound = features.tail(2)
                elif preprocess == 'MinMaxScaler':
                    features, scaler = normalize(data, 'MinMaxScaler')
                    vars_bound = features.tail(2)

            if not len(inputs['lb']) == len(inputs['ub']) == inputs['n dim']:
                st.warning('the variable number should be %d' % vars_bound.shape[1])
            else:
                st.info("the variable number is correct")


                pareto_front = find_non_dominated_solutions(targets.values, target_selected_option)
                pareto_front = pd.DataFrame(pareto_front, columns=target_selected_option)

                if inputs['objective'] == 'max':  

                    targets = - targets
                    pareto_front = find_non_dominated_solutions(targets.values, target_selected_option)
                    pareto_front = pd.DataFrame(pareto_front, columns=target_selected_option)
                    pareto_front = -pareto_front
                    targets = -targets
    
                col1, col2 = st.columns([2, 1])
                with col1:
                    with plt.style.context(['nature','no-latex']):
                        fig, ax = plt.subplots()
                        ax.plot(pareto_front[target_selected_option[0]], pareto_front[target_selected_option[1]], 'k--')
                        ax.scatter(targets[target_selected_option[0]], targets[target_selected_option[1]])
                        ax.set_xlabel(target_selected_option[0])
                        ax.set_ylabel(target_selected_option[1])
                        ax.set_title('Pareto front of visual space')
                        st.pyplot(fig)
                with col2:
                    st.write(pareto_front)
                    tmp_download_link = download_button(pareto_front, f'Pareto_front.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)

            with st.container():
                button_train = st.button('Opt', use_container_width=True)

            if button_train:               
                plot = customPlot()  
                if inputs['model'] == 'NSGA-II':
                    alg = NSGA2(
                        pop_size=inputs['size pop'],
                        # n_offsprings=inputs['n_offsprings'],
                        crossover=nsgaSBX(prob=0.9, eta=15),
                        mutation=PM(eta=20),
                        eliminate_duplicates=True
                    )
                    termination = get_termination("n_gen", inputs['max iter'])                    
                    class MyProblem(ElementwiseProblem):
                        def __init__(self):
                            super().__init__(n_var=inputs['n dim'],
                                            n_obj=2,
                                            xl=np.array(inputs['lb']),
                                            xu=np.array(inputs['ub']))
                        def _evaluate(self, x, out, *args, **kwargs):
                            x = x.reshape(1,-1)
                            y1_pred = model_1.predict(x)
                            if inputs['objective'] == 'max':
                                y1_pred = -y1_pred
                            y2_pred = model_2.predict(x)
                            if inputs['objective'] == 'max':
                                y2_pred = -y2_pred
                            out["F"] = [y1_pred, y2_pred]
    
                    problem = MyProblem()                    
                    res = minimize(problem,
                                    alg,
                                    termination,
                                    seed=inputs['random state'],
                                    save_history=True,
                                    verbose=False)
                    res.F[:, [0, 1]] = res.F[:, [1, 0]]
                    if inputs['objective'] == 'max':
                        best_y = res.F
                        targets = - targets
                        iter_data = np.concatenate([targets.values, best_y], axis = 0)
                        iter_pareto_front = find_non_dominated_solutions(iter_data, target_selected_option)
                        iter_pareto_front = pd.DataFrame(iter_pareto_front, columns=target_selected_option)
                        iter_pareto_front = -iter_pareto_front       
                        pareto_front = find_non_dominated_solutions(targets.values, target_selected_option)
                        pareto_front = pd.DataFrame(pareto_front, columns=target_selected_option)
                        pareto_front = -pareto_front
                        targets = - targets
                        best_y = - res.F

                    else:
                        best_y = res.F
                        iter_data = np.concatenate([targets.values, best_y], axis = 0)
                        iter_pareto_front = find_non_dominated_solutions(iter_data, target_selected_option)
                        iter_pareto_front = pd.DataFrame(iter_pareto_front, columns=target_selected_option)
                    
                    with plt.style.context(['nature','no-latex']):
                        fig, ax = plt.subplots()
                        ax.plot(iter_pareto_front[target_selected_option[0]],iter_pareto_front[target_selected_option[1]], 'r--')
                        ax.plot(pareto_front[target_selected_option[0]], pareto_front[target_selected_option[1]], 'k--')
                        ax.scatter(targets[target_selected_option[0]], targets[target_selected_option[1]])
                        ax.scatter(best_y[:, 0], best_y[:,1])
                        ax.set_xlabel(target_selected_option[0])
                        ax.set_ylabel(target_selected_option[1])
                        ax.set_title('Pareto front of visual space')
                        st.pyplot(fig)                    
                    best_x = res.X

                    st.info('Recommmended Sample')
                    truncate_func = np.vectorize(lambda x: '{:,.4f}'.format(x))

                    best_x = truncate_func(best_x)
                    
                    best_x = pd.DataFrame(best_x, columns = features_name)
                    if preprocess == 'StandardScaler':
                        best_x = inverse_normalize(best_x, scaler, 'StandardScaler')
                    elif preprocess == 'MinMaxScaler':
                        best_x = inverse_normalize(best_x, scaler, 'MinMaxScaler')
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(best_x)         
                        tmp_download_link = download_button(best_x, f'recommended samples.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)
                    with col2:
                        st.write(iter_pareto_front)
                        tmp_download_link = download_button(iter_pareto_front, f'iter_pareto_front.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

    elif sub_option == "迁移学习-多目标代理优化":
        colored_header(label="迁移学习-多目标代理优化",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.pickle` model and `.csv` file",  label_visibility="collapsed", accept_multiple_files=True)
        if len(file) < 6:
            table = PrettyTable(['上传文件名称', '名称','数据说明'])
            table.add_row(['file_1','dataset','数据集'])
            table.add_row(['file_2','boundary','设计变量上下界'])
            table.add_row(['file_3','weights_1','目标1 学习器权重'])
            table.add_row(['file_4','weights_2','目标2 学习器权重'])
            table.add_row(['file_5','model_1','目标1 模型1'])
            table.add_row(['file_6','model_2','目标2 模型2'])
            st.write(table)
        if len(file) >= 6:
            df = pd.read_csv(file[0])
            check_string_NaN(df)
            colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

            target_num = st.number_input('目标变量数量',  min_value=2, max_value=10, value=2)
            
            col_feature, col_target = st.columns(2)
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())
            colored_header(label="选择目标变量", description=" ", color_name="violet-70")
            target_selected_option = st.multiselect('target', list(targets)[::-1], default=targets.columns.tolist())
            df_var = pd.read_csv(file[1])
            features_name = df_var.columns.tolist()
            range_var = df_var.values
            vars_min = get_column_min(range_var)
            vars_max = get_column_max(range_var)
            array_vars_min = np.array(vars_min).reshape(1,-1)
            array_vars_max = np.array(vars_max).reshape(1,-1)
            vars_bound = np.concatenate([array_vars_min, array_vars_max], axis=0)
            colored_header(label="特征变量范围", description=" ", color_name="violet-70")
            vars_bound = pd.DataFrame(vars_bound, columns=features_name)
            st.write(vars_bound)

            colored_header(label="Optimize", description=" ", color_name="violet-70")
            model_weight_1 = pickle.load(file[2])
            model_weight_2 = pickle.load(file[3])
            model_1 = pickle.load(file[4])
            model_2 = pickle.load(file[5])
            model_path = './models/moo'
            template_alg = model_platform(model_path)

            inputs, col2 = template_alg.show()
            inputs['lb'] = vars_min
            inputs['ub'] = vars_max
            
            if not len(inputs['lb']) == len(inputs['ub']) == inputs['n dim']:
                with col2: 
                    st.warning('the variable number should be %d' % vars_bound.shape[1])
            else:
                with col2:
                    st.info("the variable number is correct")

                pareto_front = find_non_dominated_solutions(targets.values, target_selected_option)
                pareto_front = pd.DataFrame(pareto_front, columns=target_selected_option)

                if inputs['objective'] == 'max':  

                    targets = - targets
                    pareto_front = find_non_dominated_solutions(targets.values, target_selected_option)
                    pareto_front = pd.DataFrame(pareto_front, columns=target_selected_option)
                    pareto_front = -pareto_front
                    targets = -targets
    
                col1, col2 = st.columns([2, 1])
                with col1:
                    with plt.style.context(['nature','no-latex']):
                        fig, ax = plt.subplots()
                        ax.plot(pareto_front[target_selected_option[0]], pareto_front[target_selected_option[1]], 'k--')
                        ax.scatter(targets[target_selected_option[0]], targets[target_selected_option[1]])
                        ax.set_xlabel(target_selected_option[0])
                        ax.set_ylabel(target_selected_option[1])
                        ax.set_title('Pareto front of visual space')
                        st.pyplot(fig)
                with col2:
                    st.write(pareto_front)
                    tmp_download_link = download_button(pareto_front, f'Pareto_front.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)

            with st.container():
                button_train = st.button('Opt', use_container_width=True)

            if button_train:               
                plot = customPlot()  
                if inputs['model'] == 'NSGA-II':
                    alg = NSGA2(
                        pop_size=inputs['size pop'],
                        # n_offsprings=inputs['n_offsprings'],
                        crossover=nsgaSBX(prob=0.9, eta=15),
                        mutation=PM(eta=20),
                        eliminate_duplicates=True
                    )
                    termination = get_termination("n_gen", inputs['max iter']) 
                    TrAdaboostR2_1 = TrAdaboostR2()  
                    TrAdaboostR2_1.estimator_weight = model_weight_1
                    TrAdaboostR2_1.estimators = model_1
                    TrAdaboostR2_1.N = len(model_1)   
                    TrAdaboostR2_2 = TrAdaboostR2()  
                    TrAdaboostR2_2.estimator_weight = model_weight_2
                    TrAdaboostR2_2.estimators = model_2
                    TrAdaboostR2_2.N = len(model_2)                    
                    class MyProblem(ElementwiseProblem):
                        def __init__(self):
                            super().__init__(n_var=inputs['n dim'],
                                            n_obj=2,
                                            xl=np.array(inputs['lb']),
                                            xu=np.array(inputs['ub']))
                        def _evaluate(self, x, out, *args, **kwargs):
                            x = x.reshape(1,-1)
                            y1_pred = TrAdaboostR2_1.inference(x)
                            if inputs['objective'] == 'max':
                                y1_pred = -y1_pred
                            y2_pred = TrAdaboostR2_2.inference(x)
                            if inputs['objective'] == 'max':
                                y2_pred = -y2_pred
                            out["F"] = [y1_pred, y2_pred]
    
                    problem = MyProblem()                    
                    res = minimize(problem,
                                    alg,
                                    termination,
                                    seed=inputs['random state'],
                                    save_history=True,
                                    verbose=False)
                    res.F[:, [0, 1]] = res.F[:, [1, 0]]
                    if inputs['objective'] == 'max':
                        best_y = res.F
                        targets = - targets
                        iter_data = np.concatenate([targets.values, best_y], axis = 0)
                        iter_pareto_front = find_non_dominated_solutions(iter_data, target_selected_option)
                        iter_pareto_front = pd.DataFrame(iter_pareto_front, columns=target_selected_option)
                        iter_pareto_front = -iter_pareto_front       
                        pareto_front = find_non_dominated_solutions(targets.values, target_selected_option)
                        pareto_front = pd.DataFrame(pareto_front, columns=target_selected_option)
                        pareto_front = -pareto_front
                        targets = - targets
                        best_y = - res.F

                    else:
                        best_y = res.F
                        iter_data = np.concatenate([targets.values, best_y], axis = 0)
                        iter_pareto_front = find_non_dominated_solutions(iter_data, target_selected_option)
                        iter_pareto_front = pd.DataFrame(iter_pareto_front, columns=target_selected_option)
                    
                    with plt.style.context(['nature','no-latex']):
                        fig, ax = plt.subplots()
                        ax.plot(iter_pareto_front[target_selected_option[0]],iter_pareto_front[target_selected_option[1]], 'r--')
                        ax.plot(pareto_front[target_selected_option[0]], pareto_front[target_selected_option[1]], 'k--')
                        ax.scatter(targets[target_selected_option[0]], targets[target_selected_option[1]])
                        ax.scatter(best_y[:, 0], best_y[:,1])
                        ax.set_xlabel(target_selected_option[0])
                        ax.set_ylabel(target_selected_option[1])
                        ax.set_title('Pareto front of visual space')
                        st.pyplot(fig)                    
                    best_x = res.X

                    st.info('Recommmended Sample')
                    truncate_func = np.vectorize(lambda x: '{:,.4f}'.format(x))

                    best_x = truncate_func(best_x)
                    
                    best_x = pd.DataFrame(best_x, columns = features_name)
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(best_x)         
                        tmp_download_link = download_button(best_x, f'recommended samples.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)
                    with col2:
                        st.write(iter_pareto_front)
                        tmp_download_link = download_button(iter_pareto_front, f'iter_pareto_front.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

    elif sub_option == "迁移学习-单目标代理优化":
        colored_header(label="迁移学习-单目标代理优化",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.pickle` model and `.csv` file",  label_visibility="collapsed", accept_multiple_files=True)
        if len(file) < 3:
            table = PrettyTable(['上传文件名称', '名称','数据说明'])
            table.add_row(['file_1','boundary','设计变量上下界'])
            table.add_row(['file_2','weights','学习器权重'])
            table.add_row(['file_3','model','模型'])
            st.write(table)
        elif len(file) >= 3:

            df_var = pd.read_csv(file[0])
            features_name = df_var.columns.tolist()
            range_var = df_var.values
            vars_min = get_column_min(range_var)
            vars_max = get_column_max(range_var)
            array_vars_min = np.array(vars_min).reshape(1,-1)
            array_vars_max = np.array(vars_max).reshape(1,-1)
            vars_bound = np.concatenate([array_vars_min, array_vars_max], axis=0)
            colored_header(label="特征变量范围", description=" ", color_name="violet-70")
            vars_bound = pd.DataFrame(vars_bound, columns=features_name)
            st.write(vars_bound)
        
            model_weight = pickle.load(file[1])
            colored_header(label="Optimize", description=" ", color_name="violet-70")
            model = pickle.load(file[2])
            model_path = './models/surrogate optimize'
            template_alg = model_platform(model_path)

            inputs, col2 = template_alg.show()
            inputs['lb'] = vars_min
            inputs['ub'] = vars_max

            if not len(inputs['lb']) == len(inputs['ub']) == inputs['n dim']:
                st.warning('the variable number should be %d' % vars_bound.shape[1])
            else:
                st.info("the variable number is correct")

            with st.container():
                button_train = st.button('Opt', use_container_width=True)
            if button_train:  
                TrAdaboostR2 = TrAdaboostR2()
                TrAdaboostR2.estimator_weight = model_weight
                TrAdaboostR2.estimators = model
                TrAdaboostR2.N = len(model)
                def opt_func(x):
                    x = x.reshape(1,-1)
                    y_pred = TrAdaboostR2.inference(x)
                    if inputs['objective'] == 'max':
                        y_pred = -y_pred
                    return y_pred
                plot = customPlot()      
                if inputs['model'] == 'PSO':    
                    
                    alg = PSO(func=opt_func, dim=inputs['n dim'], pop=inputs['size pop'], max_iter=inputs['max iter'], lb=inputs['lb'], ub=inputs['ub'],
                            w=inputs['w'], c1=inputs['c1'], c2=inputs['c2'])
            
                    alg.run()
                    best_x = alg.gbest_x
                    best_y = alg.gbest_y

                    loss_history = alg.gbest_y_hist
                    if inputs['objective'] == 'max':
                        loss_history = -np.array(loss_history)                    

                    st.info('Recommmended Sample')
                    truncate_func = np.vectorize(lambda x: '{:,.4f}'.format(x))
                    best_x = truncate_func(best_x).reshape(1,-1)

                    best_x = pd.DataFrame(best_x, columns = features_name)

                    st.write(best_x)         
                    tmp_download_link = download_button(best_x, f'recommended samples.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                    if inputs['objective'] == 'max':
                        best_= -best_y
                    st.info('PSO best_y: %s' % best_y.item())
                    plot.evolutionary_history(loss_history, 'PSO')
                    loss_history = pd.DataFrame(loss_history)
                    tmp_download_link = download_button(loss_history, f'evolutionary history.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                elif inputs['model'] == 'GA':

                    alg = GA(pop_size=inputs['size pop'], 
                            crossover=SBX(prob=0.9, eta=15),
                            mutation=PM(eta=20),
                            eliminate_duplicates=True)

                    termination = get_termination("n_gen", inputs['max iter'])                    
                    class MyProblem(ElementwiseProblem):
                        def __init__(self):
                            super().__init__(n_var=inputs['n dim'],
                                            n_obj=1,
                                            xl=np.array(inputs['lb']),
                                            xu=np.array(inputs['ub']))
                        def _evaluate(self, x, out, *args, **kwargs):
                            x = x.reshape(1,-1)
                            y_pred = model.predict(x)
                            if inputs['objective'] == 'max':
                                y_pred = -y_pred
                            out["F"] = y_pred
                            
                    problem = MyProblem()                    
                    res = minimize(problem,
                                    alg,
                                    termination,
                                    seed=1,
                                    save_history=True,
                                    verbose=False)
                    if inputs['objective'] == 'max':
                        best_y = -res.F
                    else:
                        best_y = res.F
                    best_x = res.X
                    hist = res.history
                    hist_F = []              # the objective space values in each generation
                    # st.write("Best solution found: \nX = %s\nF = %s" % (best_x, y_pred))
                    for algo in hist:
                        # retrieve the optimum from the algorithm
                        opt = algo.opt
                        # filter out only the feasible and append and objective space values
                        feas = np.where(opt.get("feasible"))[0]
                        hist_F.append(opt.get("F")[feas])
                    # replace this line by `hist_cv` if you like to analyze the least feasible optimal solution and not the population
                    if inputs['objective'] == 'max':
                        loss_history = - np.array(hist_F).reshape(-1,1)
                    else:
                        loss_history = np.array(hist_F).reshape(-1,1)
                    st.info('Recommmended Sample')
                    truncate_func = np.vectorize(lambda x: '{:,.4f}'.format(x))
                    best_x = truncate_func(best_x).reshape(1,-1)
                    
                    best_x = pd.DataFrame(best_x, columns = features_name)

                    st.write(best_x)         
                    tmp_download_link = download_button(best_x, f'recommended samples.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                    st.info('GA best_y: %s' %  best_y.item())
                    plot.evolutionary_history(loss_history, 'GA')   
                    loss_history = pd.DataFrame(loss_history)
                    tmp_download_link = download_button(loss_history, f'evolutionary history.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                elif inputs['model'] == 'DE':
                    alg = DE(func=opt_func, n_dim=inputs['n dim'], size_pop=inputs['size pop'], max_iter=inputs['max iter'], lb=inputs['lb'], ub=inputs['ub'],
                            prob_mut = inputs['prob mut'], F=inputs['F'])

                    best_x, best_y = alg.run()

                    loss_history = alg.generation_best_Y
                    if inputs['objective'] == 'max':
                        loss_history = -np.array(loss_history)     

                    st.info('Recommmended Sample')
                    truncate_func = np.vectorize(lambda x: '{:,.4f}'.format(x))
                    best_x = truncate_func(best_x).reshape(1,-1)
                    

                    best_x = pd.DataFrame(best_x, columns = features_name)


                    st.write(best_x)         
                    tmp_download_link = download_button(best_x, f'recommended samples.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                    if inputs['objective'] == 'max':
                        best_y = -best_y               
                    st.info('DE best_y: %s' %  best_y.item())    
                    plot.evolutionary_history(loss_history, 'DE')  
                    loss_history = pd.DataFrame(loss_history)
                    tmp_download_link = download_button(loss_history, f'evolutionary history.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                elif inputs['model'] == 'SA':
                    x0 = calculate_mean(inputs['lb'], inputs['ub'])
                    alg = SAFast(func=opt_func, x0=x0, T_max = inputs['T max'], q=inputs['q'], L=inputs['L'], max_stay_counter=inputs['max stay counter'],
                                lb=inputs['lb'], ub=inputs['ub'])

                    best_x, best_y = alg.run()

                    loss_history = alg.generation_best_Y
                    if inputs['objective'] == 'max':
                        loss_history = -np.array(loss_history)     

                    st.info('Recommmended Sample')
                    truncate_func = np.vectorize(lambda x: '{:,.4f}'.format(x))
                    best_x = truncate_func(best_x).reshape(1,-1)

                    best_x = pd.DataFrame(best_x, columns = features_name)
                    st.write(best_x)         
                    tmp_download_link = download_button(best_x, f'recommended samples.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                    if inputs['objective'] == 'max':
                        best_y = -best_y  
                    st.info('SA best_y: %s' %  best_y.item()) 
                    plot.evolutionary_history(loss_history, 'SA')  

                    loss_history = pd.DataFrame(loss_history)
                    tmp_download_link = download_button(loss_history, f'evolutionary history.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                    
elif select_option == "其他":
    with st.sidebar:
        sub_option = option_menu(None, ["模型推理","可解释性机器学习"])
    if sub_option == "可解释性机器学习":
        colored_header(label="可解释性机器学习",description=" ",color_name="violet-90")

        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
        if file is None:
            table = PrettyTable(['上传文件名称', '名称','数据说明'])
            table.add_row(['file_1','dataset','数据集'])
            st.write(table)        
        if file is not None:
            df = pd.read_csv(file)
            # 检测缺失值
            check_string_NaN(df)
            colored_header(label="数据信息", description=" ",color_name="violet-70")
            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

            target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)
            
            col_feature, col_target = st.columns(2)
                
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())

            colored_header(label="Shapley value",description=" ",color_name="violet-70")

            fs = FeatureSelector(features, targets)

            target_selected_option = st.selectbox('choose target', list(fs.targets))
            fs.targets = fs.targets[target_selected_option]

            reg = RFR()
            X_train, X_test, y_train, y_test = TTS(fs.features, fs.targets, random_state=0) 
            test_size = st.slider('test size',0.1, 0.5, 0.2) 
            random_state = st.checkbox('random state 1024',True)
            if random_state:
                random_state = 42
            else:
                random_state = None
                
            fs.Xtrain,fs.Xtest, fs.Ytrain, fs.Ytest = TTS(fs.features,fs.targets,test_size=test_size,random_state=random_state)
            reg.fit(fs.Xtrain, fs.Ytrain)

            explainer = shap.TreeExplainer(reg)
            
            shap_values = explainer(fs.features)

            colored_header(label="SHAP Feature Importance", description=" ",color_name="violet-30")
            nfeatures = st.slider("features", 2, fs.features.shape[1],fs.features.shape[1])
            st_shap(shap.plots.bar(shap_values, max_display=nfeatures))

            colored_header(label="SHAP Feature Cluster", description=" ",color_name="violet-30")
            clustering = shap.utils.hclust(fs.features, fs.targets)
            clustering_cutoff = st.slider('clustering cutoff', 0.0,1.0,0.5)
            nfeatures = st.slider("features", 2, fs.features.shape[1],fs.features.shape[1], key=2)
            st_shap(shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=clustering_cutoff, max_display=nfeatures))

            colored_header(label="SHAP Beeswarm", description=" ",color_name="violet-30")
            rank_option = st.selectbox('rank option',['max','mean'])
            max_dispaly = st.slider('max display',2, fs.features.shape[1],fs.features.shape[1])
            if rank_option == 'max':
                st_shap(shap.plots.beeswarm(shap_values, order = shap_values.abs.max(0), max_display =max_dispaly))
            else:
                st_shap(shap.plots.beeswarm(shap_values, order = shap_values.abs.mean(0), max_display =max_dispaly))

            colored_header(label="SHAP Dependence", description=" ",color_name="violet-30")
            
            shap_values = explainer.shap_values(fs.features) 
            list_features = fs.features.columns.tolist()
            feature = st.selectbox('feature',list_features)
            interact_feature = st.selectbox('interact feature', list_features)
            st_shap(shap.dependence_plot(feature, shap_values, fs.features, display_features=fs.features,interaction_index=interact_feature))

    elif sub_option == "模型推理":
        
        colored_header(label="模型推理",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv`file", label_visibility="collapsed", accept_multiple_files=True)
        if len(file) < 2:
            table = PrettyTable(['上传文件名称', '名称','数据说明'])
            table.add_row(['file_1','data set','数据集'])
            table.add_row(['file_2','model','模型'])
            st.write(table)
        elif len(file) == 2:
            df = pd.read_csv(file[0])
            model_file = file[1]
     
            check_string_NaN(df)

            colored_header(label="数据信息", description=" ",color_name="violet-70")
            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

            target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)

            col_feature, col_target = st.columns(2)
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())    
            colored_header(label="选择目标变量", description=" ", color_name="violet-70")

            target_selected_option = st.selectbox('target', list(targets)[::-1])

            targets = targets[target_selected_option]
            preprocess = st.selectbox('data preprocess',[None, 'StandardScaler','MinMaxScaler'])
            if preprocess == 'StandardScaler':
                features = StandardScaler().fit_transform(features)
            elif preprocess == 'MinMaxScaler':
                features = MinMaxScaler().fit_transform(features)

            model = pickle.load(model_file)
            prediction = model.predict(features)

            plot = customPlot()
            plot.pred_vs_actual(targets, prediction)
            r2 = r2_score(targets, prediction)
            st.write('R2: {}'.format(r2))
            result_data = pd.concat([targets, pd.DataFrame(prediction)], axis=1)
            result_data.columns = ['actual','prediction']
            with st.expander('预测结果'):
                st.write(result_data)
                tmp_download_link = download_button(result_data, f'预测结果.csv', button_text='download')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
            st.write('---')

