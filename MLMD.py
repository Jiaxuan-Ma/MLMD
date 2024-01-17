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
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RationalQuadratic, CompoundKernel, \
Exponentiation,ConstantKernel, ExpSineSquared, Hyperparameter, Kernel, Matern, PairwiseKernel, Product, RationalQuadratic, RBF, Sum
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
from bayes_opt import BayesianOptimization

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
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling

from sko.PSO import PSO
from sko.DE import DE
# from sko.AFSA import AFSA
from sko.SA import SAFast, SABoltzmann

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
    **Contact**: 

    terry.jx.ma@gmail.com (Jiaxuan Ma)
    george-jie.xiong@connect.polyu.hk (Jie Xiong)         

''')
    select_option = option_menu("MLMD", ["Home Page", "Basic Data", "Feature Engineering","Cluster & ReduceDim", "Regression", "Classification", "Transfer Learning", "Model Inference","Surrogate Optimization","Active Learning","Interpretable Machine Learning"],
                    icons=['house', 'clipboard-data', 'menu-button-wide','circle','bezier2', 'subtract', 'arrow-repeat', 'app', 'microsoft','search','book-half'],
                    menu_icon="boxes", default_index=0)
if select_option == "Home Page":
    st.write('''![](https://user-images.githubusercontent.com/61132191/231174459-96d33cdf-9f6f-4296-ba9f-31d11056ef12.jpg?raw=true)''')


    colored_header(label="Machine Learning for Material Design",description="MLMD is dedicated to the integration of material experiment and material design, and accelerate the new material discovery with desired one or more properties.",color_name="violet-90")

    colored_header(label="Data Layout",description="only support `.csv` file",color_name="violet-90")

    st.write('''![](https://github.com/Jiaxuan-Ma/MLMD/assets/61132191/4733175e-1d24-46a1-8bf7-6a01fb284af4?raw=true)''')
    st.write(
        '''In order to ensure the sum of mass fraction of all compositions satisfa 100 \% in optimization, the base element column needs to be removed when using the **surrogate optimization module**.''')
    colored_header(label="Acknowledgements",description="",color_name="violet-90")

    st.markdown(
    '''
    #### Contributors
    **Research Group**: Materials and mechanics informatics lab (MMIL), Shanghai Unversity

    **Developers**: Jiaxuan Ma (PhD Candidate), Bin Cao (PhD Candidate), Yuan Tian (Doctor), Jie Xiong (Assist Prof), Sheng Sun (Prof)

    #### Funding
    This work was nancially supported by the National Key Research and Development
    Program of China (No. 2022YFB3707803), the National Natural Science Foundation of
    China Project (No. 12072179 and 11672168), the Key Research Project of Zhejiang Lab
    (No. 2021PE0AC02), Shanghai Pujiang Program Grant (23PJ1403500), and Shanghai
    Engineering Research Center for Integrated Circuits and Advanced Display Materials.
    ''')

elif select_option == "Basic Data":
    with st.sidebar:
        sub_option = option_menu(None, ["Databases", "Data visualization"])
    if sub_option == "Databases":
        colored_header(label="Databases",description=" ",color_name="violet-90")
        col1, col2 = st.columns([2,2])
        with col1:
            df = pd.read_csv('./data/Case 1.csv')
            st.write("Polycrystalline ceramic")
            image = Image.open('./data/fig4.png')
            st.image(image, width=280,caption='')
            tmp_download_link = download_button(df, f'data.csv', button_text='download')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
        with col2:
            df = pd.read_csv('./data/Case 3_R.csv')
            st.write("FGH98 superalloy")
            image = Image.open('./data/fig6.png')
            st.image(image, width=280,caption='')
            tmp_download_link = download_button(df , f'data.csv', button_text='download')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

        col1, col2 = st.columns([2,2])
        with col1:
            df = pd.read_csv('./data/HEA-Dataset.csv')
            st.write("High entropy alloy")
            image = Image.open('./data/fig3.png')
            st.image(image, width=280,caption='')
            tmp_download_link = download_button(df , f'data.csv', button_text='download')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

        with col2:
            df = pd.read_csv('./data/Case 1_R.csv')
            st.write("Low-alloy steel")
            image = Image.open('./data/fig1.png')
            st.image(image, width=280,caption='')
            tmp_download_link = download_button(df , f'data.csv', button_text='download')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

        col1, col2 = st.columns([2,2])
        with col1:
            df = pd.read_csv('./data/RAFM-dataset.csv')
            st.write("Ferriticmartensitic steel")
            image = Image.open('./data/fig5.png')
            st.image(image, width=280,caption='')
            tmp_download_link = download_button(df , f'data.csv', button_text='download')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
        with col2:
            df = pd.read_csv('./data/Case 2.csv')
            st.write("Amorphous Alloy")
            image = Image.open('./data/fig2.png')
            st.image(image, width=280,caption='')
            tmp_download_link = download_button(df , f'data.csv', button_text='download')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

    elif sub_option == "Data visualization":

        colored_header(label="Data Visualization",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv` file", type=['csv'], label_visibility="collapsed")
        if file is None:
            table = PrettyTable(['file name', 'class','description'])
            table.add_row(['file_1','dataset','data file'])
            st.write(table)
        if file is not None:
            df = pd.read_csv(file)
            check_string_NaN(df)
            
            colored_header(label="Data information",description=" ",color_name="violet-70")

            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="Data statistics",description=" ",color_name="violet-30")

            st.write(df.describe())

            tmp_download_link = download_button(df.describe(), f'statistics.csv', button_text='download')
            
            st.markdown(tmp_download_link, unsafe_allow_html=True)

            colored_header(label="Feature and target", description=" ",color_name="violet-70")
            
            target_num = st.number_input('target number', min_value=1, max_value=10, value=1)
            col_feature, col_target = st.columns(2)
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())

            colored_header(label="Feature distribution", description=" ",color_name="violet-30")
            feature_selected_name = st.selectbox('feature', list(features),1)
            feature_selected_value = features[feature_selected_name]
            plot = customPlot()
            col1, col2 = st.columns([1,3])
            with col1:  
                with st.expander("plot parameters"):
                    options_selected = [plot.set_title_fontsize(18),plot.set_label_fontsize(19),
                                plot.set_tick_fontsize(20),plot.set_legend_fontsize(21), plot.set_color('bin color', 0, 22)]
            with col2:
                plot.feature_distribution(options_selected,feature_selected_name,feature_selected_value)
            
            with col1:  
                with st.expander("plot parameters"):
                    options_selected = [plot.set_title_fontsize(1),plot.set_label_fontsize(2),
                                plot.set_tick_fontsize(3),plot.set_legend_fontsize(4),plot.set_color('line color',6,5),plot.set_color('bin color',0,6)]
            with col2:
                plot.feature_hist_kde(options_selected,feature_selected_name,feature_selected_value)

            #=========== Targets visulization ==================

            colored_header(label="Target distribution", description=" ",color_name="violet-30")

            target_selected_name = st.selectbox('target',list(targets))

            target_selected_value = targets[target_selected_name]
            plot = customPlot()
            col1, col2 = st.columns([1,3])
            with col1:  
                with st.expander("plot parameters"):
                    options_selected = [plot.set_title_fontsize(7),plot.set_label_fontsize(8),
                                plot.set_tick_fontsize(9),plot.set_legend_fontsize(10), plot.set_color('line color',6,11), plot.set_color('bin color',0,12)]
            with col2:
                plot.target_hist_kde(options_selected,target_selected_name,target_selected_value)

            #=========== Features analysis ==================

            colored_header(label="Recipes of feature ", description=" ",color_name="violet-30")

            feature_range_selected_name = st.slider('feature number',1,len(features.columns), (1,2))
            min_feature_selected = feature_range_selected_name[0]-1
            max_feature_selected = feature_range_selected_name[1]
            feature_range_selected_value = features.iloc[:,min_feature_selected: max_feature_selected]
            data_by_feature_type = df.groupby(list(feature_range_selected_value))
            feature_type_data = create_data_with_group_and_counts(data_by_feature_type)
            IDs = [str(id_) for id_ in feature_type_data['ID']]
            Counts = feature_type_data['Count']
            col1, col2 = st.columns([1,3])
            with col1:  
                with st.expander("plot parameters"):
                    options_selected = [plot.set_title_fontsize(13),plot.set_label_fontsize(14),
                                plot.set_tick_fontsize(15),plot.set_legend_fontsize(16),plot.set_color('bin color',0, 17)]
            with col2:
                plot.featureSets_statistics_hist(options_selected,IDs, Counts)
        st.write('---')

elif select_option == "Feature Engineering":
    with st.sidebar:
        sub_option = option_menu(None, ["Missing Value","Feature Transform", "Duplicate Value", "Feature Correlation", "Feature & Target Correlation", "One-hot Coding", "Feature Importance Rank"])

    if sub_option == "Missing Value":
        colored_header(label="Missing Value",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
        if file is None:
            table = PrettyTable(['file name', 'class','description'])
            table.add_row(['file_1','dataset','data file'])
            st.write(table)
        if file is not None:
            df = pd.read_csv(file)
            null_columns = df.columns[df.isnull().any()]
            if len(null_columns) == 0:
                st.error('No missing features!')
                st.stop()
                
            colored_header(label="Data information", description=" ",color_name="violet-70")
            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="Feature and target",description=" ",color_name="violet-70")

            target_num = st.number_input('target number',  min_value=1, max_value=10, value=1)
            
            col_feature, col_target = st.columns(2)
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())
        
            colored_header(label="method",description=" ",color_name="violet-70")
            sub_sub_option = option_menu(None, ["drop missing value", "fill missing value"],
                    icons=['house',  "list-task"],
                    menu_icon="cast", default_index=0, orientation="horizontal")
            if sub_sub_option == "drop missing value":
                fs = FeatureSelector(features, targets)
                missing_threshold = st.slider("drop threshold",0.001, 1.0, 0.5)
                fs.identify_missing(missing_threshold)
                fs.features_dropped_missing = fs.features.drop(columns=fs.ops['missing'])
                
                data = pd.concat([fs.features_dropped_missing, targets], axis=1)
                st.write(data)
                tmp_download_link = download_button(data, f'dropeddata.csv', button_text='download')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write('%d features with $\gt$ %0.2f missing threshold.\n' % (len(fs.ops['missing']), fs.missing_threshold))
                plot = customPlot()

                with st.expander('plot parameters'):
                    col1, col2 = st.columns([1,3])
                    with col1:
                        options_selected = [plot.set_title_fontsize(1),plot.set_label_fontsize(2),
                                    plot.set_tick_fontsize(3),plot.set_legend_fontsize(4),plot.set_color('bin color',19,5)]
                    with col2:
                        plot.feature_missing(options_selected, fs.record_missing, fs.missing_stats)
                st.write('---')
            if sub_sub_option == "fill missing value":
                fs = FeatureSelector(features, targets)
                missing_feature_list = fs.features.columns[fs.features.isnull().any()].tolist()
                with st.container():
                    fill_method = st.selectbox('fill method',('constant', 'random forest'))
                
                if fill_method == 'constant':

                    missing_feature = st.multiselect('feature of drop value',missing_feature_list,missing_feature_list[-1])
                    
                    option_filled = st.selectbox('mean',('mean','constant','median','mode'))
                    if option_filled == 'mean':
                        # fs.features[missing_feature] = fs.features[missing_feature].fillna(fs.features[missing_feature].mean())
                        imp = SimpleImputer(missing_values=np.nan,strategy= 'mean')

                        fs.features[missing_feature] = imp.fit_transform(fs.features[missing_feature])
                    elif option_filled == 'constant':
                        # fs.features[missing_feature] = fs.features[missing_feature].fillna(0)
                        fill_value = st.number_input('value')
                        imp = SimpleImputer(missing_values=np.nan, strategy= 'constant', fill_value = fill_value)
                        
                        fs.features[missing_feature] = imp.fit_transform(fs.features[missing_feature])
                    elif option_filled == 'median':
                        # fs.features[missing_feature] = fs.features[missing_feature].fillna(0)
                        imp = SimpleImputer(missing_values=np.nan, strategy= 'median')
                        
                        fs.features[missing_feature] = imp.fit_transform(fs.features[missing_feature])
                    elif option_filled == 'mode':

                        imp = SimpleImputer(missing_values=np.nan, strategy= 'most_frequent')
                        
                        fs.features[missing_feature] = imp.fit_transform(fs.features[missing_feature])  

                    data = pd.concat([fs.features, targets], axis=1)
                else:
                    with st.expander('hyper parameters'):
                        num_estimators = st.number_input('number estimators',1, 10000, 100)
                        criterion = st.selectbox('criterion',('squared_error','absolute_error','friedman_mse','poisson'))
                        max_depth = st.number_input('max depth',1, 1000, 5)
                        min_samples_leaf = st.number_input('min samples leaf', 1, 1000, 5)
                        min_samples_split = st.number_input('min samples split', 1, 1000, 5)
                        random_state = st.checkbox('random state 1024',True)


                    option_filled = st.selectbox('mean',('mean','constant','median','mode'))
                    if option_filled == 'mean':
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

                    elif option_filled == 'constant':

                        fill_value = st.number_input('value')
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
                        
                    elif option_filled == 'median':
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
            
                    elif option_filled == 'mode':

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

                tmp_download_link = download_button(data, f'fillmissing.csv', button_text='download')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write('---')
    
    elif sub_option == "Duplicate Value":
        colored_header(label="Duplicate Value",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
        if file is None:
            table = PrettyTable(['file name', 'class','description'])
            table.add_row(['file_1','dataset','data file'])
            st.write(table)
        if file is not None:
            colored_header(label="Data information",description=" ",color_name="violet-70")
            df = pd.read_csv(file)
            check_string_NaN(df)

            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="Feature and target",description=" ",color_name="violet-70")

            target_num = st.number_input('target number',  min_value=1, max_value=10, value=1)
            
            col_feature, col_target = st.columns(2)
            
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())

            colored_header(label="Feature of drop duplicate value",description=" ",color_name="violet-70")
            fs = FeatureSelector(features, targets)
            plot = customPlot() 

            col1, col2 = st.columns([1,3])
            with col1:
                
                fs.identify_nunique()
                option_counts = st.slider('number of drop duplicate value',0, int(fs.unique_stats.max())-1,1)
                st.write(fs.unique_stats)
            with col2:

                fs.identify_nunique(option_counts)
                fs.features_dropped_single = fs.features.drop(columns=fs.ops['single_unique'])
                data = pd.concat([fs.features_dropped_single, targets], axis=1)
                st.write(fs.features_dropped_single)
                
                tmp_download_link = download_button(data, f'dropduplicate.csv', button_text='download')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write('%d features $\leq$  %d unique value.\n' % (len(fs.ops['single_unique']),option_counts))
    
            with st.expander('plot parameters'):
                col1, col2 = st.columns([1,3])
                with col1:
                    options_selected = [plot.set_title_fontsize(6),plot.set_label_fontsize(7),
                                plot.set_tick_fontsize(8),plot.set_legend_fontsize(9),plot.set_color('bin color',19,10)]
                with col2:
                    plot.feature_nunique(options_selected, fs.record_single_unique,fs.unique_stats)     
                
            st.write('---')

    elif sub_option == "Feature Transform":
        colored_header(label="Feature Transform",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
        if file is None:
            table = PrettyTable(['Composition'])
            table.add_row(['Ti50Cu42.5Ni7.5'])
            st.write(table)
        if file is not None:
            colored_header(label="Data information",description=" ",color_name="violet-70")

            df = pd.read_csv(file)
            df_nrow = df.head()
            st.write(df_nrow)
            option = st.selectbox('option',['Alloy', 'Inorganic'])
            button = st.button('Transform', use_container_width=True)
            if button:
                df = feature_transform(df, option)
                st.write(df.head())
                tmp_download_link = download_button(df, f'trans_data.csv', button_text='download')
                st.markdown(tmp_download_link, unsafe_allow_html=True)      

    elif sub_option == "Feature Correlation":
        colored_header(label="Feature Correlation",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
        if file is None:
            table = PrettyTable(['file name', 'class','description'])
            table.add_row(['file_1','dataset','data file'])
            st.write(table)
        if file is not None:
            df = pd.read_csv(file)
            check_string_NaN(df)
            colored_header(label="Data information", description=" ",color_name="violet-70")
            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="Feature and target",description=" ",color_name="violet-70")

            target_num = st.number_input('target number',  min_value=1, max_value=10, value=1)
            
            col_feature, col_target = st.columns(2)
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())
        
            colored_header(label="Drop collinear feature",description=" ",color_name="violet-30")
            fs = FeatureSelector(features, targets)
            plot = customPlot() 

            target_selected_option = st.selectbox('target', list(fs.targets))
            target_selected = fs.targets[target_selected_option]

            col1, col2 = st.columns([1,3])
            with col1:
                corr_method = st.selectbox("correlation analysis method",["pearson","spearman","kendall"])
                correlation_threshold = st.slider("correlation threshold",0.001, 1.0, 0.9) 
                corr_matrix = pd.concat([fs.features, target_selected], axis=1).corr(corr_method)
                fs.identify_collinear(corr_matrix, correlation_threshold)
                fs.judge_drop_f_t_after_f_f([target_selected_option], corr_matrix)

                is_mask = st.selectbox('mask',('Yes', 'No'))
                with st.expander('plot parameters'):
                    options_selected = [plot.set_tick_fontsize(21), plot.set_tick_fontsize(22)]
                with st.expander('collinear feature'):
                    st.write(fs.record_collinear)
            with col2:
                fs.features_dropped_collinear = fs.features.drop(columns=fs.ops['collinear'])
                assert fs.features_dropped_collinear.size != 0,'zero feature !' 
                corr_matrix_drop_collinear = fs.features_dropped_collinear.corr(corr_method)
                plot.corr_cofficient(options_selected, is_mask, corr_matrix_drop_collinear)
                with st.expander('dropped data'):
                    data = pd.concat([fs.features_dropped_collinear, targets], axis=1)
                    st.write(data)
                    tmp_download_link = download_button(data, f'droppedcollinear.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)

    elif sub_option == "Feature & Target Correlation":
        colored_header(label="Feature & Target Correlation",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
        if file is None:
            table = PrettyTable(['file name', 'class','description'])
            table.add_row(['file_1','dataset','data file'])
            st.write(table)
        if file is not None:
            df = pd.read_csv(file)
            # 检测缺失值
            check_string_NaN(df)
            colored_header(label="Data information", description=" ",color_name="violet-70")
            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="Feature and target",description=" ",color_name="violet-70")

            target_num = st.number_input('target number',  min_value=1, max_value=10, value=1)
            
            col_feature, col_target = st.columns(2)
            
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())
        
            colored_header(label="Drop low correlation feature",description=" ",color_name="violet-70")
            fs = FeatureSelector(features, targets)
            plot = customPlot() 
            target_selected_option = st.selectbox('feature', list(fs.targets))
            col1, col2 = st.columns([1,3])
            
            with col1:  
                corr_method = st.selectbox("correlation analysis method",["pearson","spearman","kendall","MIR"], key=15)  
                if corr_method != "MIR":
                    option_dropped_threshold = st.slider('correlation threshold',0.0, 1.0,0.0)
                if corr_method == 'MIR':
                    options_seed = st.checkbox('random state 1024',True)
                with st.expander('plot parameters'):
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
                    with st.expander('dropped data'):
                        data = pd.concat([fs.features_dropped_f_t, targets], axis=1)
                        st.write(data)
                        tmp_download_link = download_button(data, f'droplowcorr.csv', button_text='download')
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

    elif sub_option == "One-hot Coding":
        colored_header(label="One-hot Coding",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
        if file is None:
            table = PrettyTable(['file name', 'class','description'])
            table.add_row(['file_1','dataset','data file'])
            st.write(table)
        if file is not None:
            df = pd.read_csv(file)
            check_string_NaN(df)
            colored_header(label="Data information", description=" ",color_name="violet-70")
            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="Feature and Target",description=" ",color_name="violet-70")

            target_num = st.number_input('target number',  min_value=1, max_value=10, value=1)
            
            col_feature, col_target = st.columns(2)
            
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())
    
            fs = FeatureSelector(features, targets)
            plot = customPlot() 
            str_col_list = fs.features.select_dtypes(include=['object']).columns.tolist()
            fs.one_hot_feature_encoder(True)
            data = pd.concat([fs.features_plus_oneHot, targets], axis=1)
            # delete origin string columns
            data = data.drop(str_col_list, axis=1)
            st.write(data)
            tmp_download_link = download_button(data, f'one-hotcoding.csv', button_text='download')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
            st.write('---')
    
    elif sub_option == "Feature Importance Rank":
        colored_header(label="Feature Importance Rank",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
        if file is None:
            table = PrettyTable(['file name', 'class','description'])
            table.add_row(['file_1','dataset','data file'])
            st.write(table)
        if file is not None:
            df = pd.read_csv(file)
            # 检测缺失值
            check_string_NaN(df)
            colored_header(label="Data information", description=" ",color_name="violet-70")
            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="Feature and target",description=" ",color_name="violet-70")

            target_num = st.number_input('target number',  min_value=1, max_value=10, value=1)
            
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

            colored_header(label="target", description=" ", color_name="violet-70")

            target_selected_name = st.selectbox('target', list(fs.targets)[::-1])

            fs.targets = targets[target_selected_name]
            
            colored_header(label="Selector", description=" ",color_name="violet-70")

            model_path = './models/feature importance'
            
            template_alg = model_platform(model_path=model_path)

            colored_header(label="Training", description=" ",color_name="violet-70")

            inputs, col2 = template_alg.show()

            if inputs['model'] == 'LinearRegressor':
                
                fs.model = LinearR()

                with col2:
                    option_cumulative_importance = st.slider('cumulative importance threshold',0.0, 1.0, 0.95)
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
                    option_cumulative_importance = st.slider('cumulative importance threshold',0.0, 1.0, 0.95)
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
                    option_cumulative_importance = st.slider('cumulative importance threshold',0.0, 1.0, 0.95)
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
                    option_cumulative_importance = st.slider('cumulative importance threshold',0.0, 1.0, 0.95)
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
                                                        min_samples_split=inputs['min samples split'],warm_start=inputs['warm start'],
                                                        n_jobs=inputs['njobs'])
                        with col2:
                            option_cumulative_importance = st.slider('cumulative importance threshold',0.5, 1.0, 0.95)
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

elif select_option == "Regression":

    colored_header(label="Regression",description=" ",color_name="violet-90")
    file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
    if file is None:
        table = PrettyTable(['file name', 'class','description'])
        table.add_row(['file_1','dataset','data file'])
        st.write(table)
    if file is not None:
        df = pd.read_csv(file)
        # 检测缺失值
        check_string_NaN(df)

        colored_header(label="Data information", description=" ",color_name="violet-70")
        nrow = st.slider("rows", 1, len(df), 5)
        df_nrow = df.head(nrow)
        st.write(df_nrow)

        colored_header(label="Feature and target",description=" ",color_name="violet-70")

        target_num = st.number_input('target number',  min_value=1, max_value=10, value=1)
        
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

        colored_header(label="target", description=" ", color_name="violet-70")

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
                    if inputs['auto hyperparameters'] == False:
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

                    elif inputs['auto hyperparameters']:
                        def DTR_TT(max_depth, min_samples_leaf, min_samples_split):
                            reg.model = tree.DecisionTreeRegressor(max_depth=int(max_depth), min_samples_leaf=int(min_samples_leaf), min_samples_split=int(min_samples_split))
                            reg.DecisionTreeRegressor()
                            return reg.score
                        
                        DTRbounds = {'max_depth':(1, inputs['max depth']), 'min_samples_leaf':(1, inputs['min samples leaf']), 'min_samples_split':(2, inputs['min samples split'])}
                        
                        with st.expander('hyperparameter opt'):
                        
                            optimizer = BayesianOptimization(f=DTR_TT, pbounds=DTRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['min_samples_leaf'] = int(params_best['min_samples_leaf'])
                        params_best['min_samples_split'] = int(params_best['min_samples_split'])
                        st.write("\n","\n","best params: ", params_best)
                        
                        reg.model = tree.DecisionTreeRegressor(random_state=inputs['random state'],splitter=inputs['splitter'],
                                        max_depth=params_best['max_depth'],min_samples_leaf=params_best['min_samples_leaf'],
                                        min_samples_split=params_best['min_samples_split']) 
                        
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
                    if inputs['auto hyperparameters'] == False:
                        reg.model = tree.DecisionTreeRegressor(random_state=inputs['random state'],splitter=inputs['splitter'],
                            max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                            min_samples_split=inputs['min samples split']) 

                        export_cross_val_results(reg, cv, "DTR_cv", inputs['random state'])

                    elif inputs['auto hyperparameters']:
                        def DTR_TT(max_depth, min_samples_leaf, min_samples_split):
                            reg.model = tree.DecisionTreeRegressor(max_depth=int(max_depth), min_samples_leaf=int(min_samples_leaf), min_samples_split=int(min_samples_split))
                            cv_score = cv_cal(reg, cv, inputs['random state'])
                            return cv_score
                        
                        DTRbounds = {'max_depth':(1, inputs['max depth']), 'min_samples_leaf':(1, inputs['min samples leaf']), 'min_samples_split':(2, inputs['min samples split'])}
                        
                        with st.expander('hyperparameter opt'):
                        
                            optimizer = BayesianOptimization(f=DTR_TT, pbounds=DTRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['min_samples_leaf'] = int(params_best['min_samples_leaf'])
                        params_best['min_samples_split'] = int(params_best['min_samples_split'])
                        st.write("\n","\n","best params: ", params_best)                        
                        reg.model = tree.DecisionTreeRegressor(random_state=inputs['random state'],splitter=inputs['splitter'],
                                        max_depth=params_best['max_depth'],min_samples_leaf=params_best['min_samples_leaf'],
                                        min_samples_split=params_best['min_samples_split']) 
                        
                        export_cross_val_results(reg, cv, "DTR_cv", inputs['random state'])                          


                elif operator == 'leave one out':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = tree.DecisionTreeRegressor(random_state=inputs['random state'],splitter=inputs['splitter'],
                            max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                            min_samples_split=inputs['min samples split']) 
                    
                        export_loo_results(reg, loo, "DTR_loo")

                    elif inputs['auto hyperparameters']:
                        def DTR_TT(max_depth, min_samples_leaf, min_samples_split):
                            reg.model = tree.DecisionTreeRegressor(max_depth=int(max_depth), min_samples_leaf=int(min_samples_leaf), min_samples_split=int(min_samples_split))
                            loo_score = loo_cal(reg, loo)
                            return loo_score
                        
                        DTRbounds = {'max_depth':(1, inputs['max depth']), 'min_samples_leaf':(1, inputs['min samples leaf']), 'min_samples_split':(2, inputs['min samples split'])}
                        
                        with st.expander('hyperparameter opt'):
                        
                            optimizer = BayesianOptimization(f=DTR_TT, pbounds=DTRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['min_samples_leaf'] = int(params_best['min_samples_leaf'])
                        params_best['min_samples_split'] = int(params_best['min_samples_split'])
                        st.write("\n","\n","best params: ", params_best)                        
                        reg.model = tree.DecisionTreeRegressor(random_state=inputs['random state'],splitter=inputs['splitter'],
                                        max_depth=params_best['max_depth'],min_samples_leaf=params_best['min_samples_leaf'],
                                        min_samples_split=params_best['min_samples_split'])                     
                        export_loo_results(reg, loo, "DTR_loo")
        if inputs['model'] == 'RandomForestRegressor':
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

                    # elif operator == 'oob score':
                    #     inputs['oob score']  = st.selectbox('oob score',[True], disabled=True)
                    #     inputs['warm start'] = True

            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            
            if button_train:
                if operator == 'train test split':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = RFR( n_estimators=inputs['nestimators'],random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                        min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                                        n_jobs=inputs['njobs'])
                        reg.RandomForestRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        plot_and_export_results(reg, "RFR")

                    elif inputs['auto hyperparameters']:
                        def RFR_TT(n_estimators, max_depth, min_samples_leaf, min_samples_split):
                            
                            reg.model = RFR(n_estimators=int(n_estimators),max_depth=int(max_depth),min_samples_leaf=int(min_samples_leaf),
                                                        min_samples_split=int(min_samples_split), n_jobs=-1)
                            reg.RandomForestRegressor()
                            return reg.score
                        
                        RFRbounds = {'n_estimators':(1, inputs['nestimators']), 'max_depth':(1, inputs['max depth']), 'min_samples_leaf':(1, inputs['min samples leaf']), 'min_samples_split':(2, inputs['min samples split'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=RFR_TT, pbounds=RFRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['min_samples_leaf'] = int(params_best['min_samples_leaf'])
                        params_best['min_samples_split'] = int(params_best['min_samples_split'])
                        st.write("\n","\n","best params: ", params_best)
                        
                        reg.model = RFR(n_estimators=params_best['n_estimators'],random_state=inputs['random state'],max_depth=params_best['max_depth'],min_samples_leaf=params_best['min_samples_leaf'],
                                    min_samples_split=params_best['min_samples_split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                    n_jobs=inputs['njobs'])
                        
                        reg.RandomForestRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        plot_and_export_results(reg, "RFR")

                elif operator == 'cross val score':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = RFR(n_estimators=inputs['nestimators'],random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                    min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                                    n_jobs=inputs['njobs'])
                        export_cross_val_results(reg, cv, "RFR_cv", inputs['random state'])
                    elif inputs['auto hyperparameters']:
                        def RFR_TT(n_estimators, max_depth, min_samples_leaf, min_samples_split):
                            reg.model = RFR(n_estimators=int(n_estimators),max_depth=int(max_depth),min_samples_leaf=int(min_samples_leaf),
                                                        min_samples_split=int(min_samples_split), n_jobs=-1)
                            cv_score = cv_cal(reg, cv, inputs['random state'])
                            return cv_score
                        
                        RFRbounds = {'n_estimators':(1, inputs['nestimators']), 'max_depth':(1, inputs['max depth']), 'min_samples_leaf':(1, inputs['min samples leaf']), 'min_samples_split':(2, inputs['min samples split'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=RFR_TT, pbounds=RFRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['min_samples_leaf'] = int(params_best['min_samples_leaf'])
                        params_best['min_samples_split'] = int(params_best['min_samples_split'])
                        st.write("\n","\n","best params: ", params_best)

                        reg.model = RFR(n_estimators=params_best['n_estimators'],random_state=inputs['random state'],max_depth=params_best['max_depth'],min_samples_leaf=params_best['min_samples_leaf'],
                                    min_samples_split=params_best['min_samples_split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                    n_jobs=inputs['njobs'])
                        
                        export_cross_val_results(reg, cv, "RFR_cv", inputs['random state'])        

                # elif operator == 'oob score':

                #     reg.model = RFR(criterion = inputs['criterion'],n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                #                                 min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                #                                 n_jobs=inputs['njobs'])
                
                #     reg_res  = reg.model.fit(reg.features, reg.targets)
                #     oob_score = reg_res.oob_score_
                #     st.write(f'oob score : {oob_score}')

                elif operator == 'leave one out':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = RFR(criterion = inputs['criterion'],n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                    min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                                    n_jobs=inputs['njobs'])
                        export_loo_results(reg, loo, "RFR_loo")
                    elif inputs['auto hyperparameters']:
                        
                        def RFR_TT(n_estimators, max_depth, min_samples_leaf, min_samples_split):
                            reg.model = RFR(n_estimators=int(n_estimators),max_depth=int(max_depth),min_samples_leaf=int(min_samples_leaf),
                                                        min_samples_split=int(min_samples_split), n_jobs=-1)
                            loo_score = loo_cal(reg, loo)
                            return loo_score
                        
                        RFRbounds = {'n_estimators':(1, inputs['nestimators']), 'max_depth':(1, inputs['max depth']), 'min_samples_leaf':(1, inputs['min samples leaf']), 'min_samples_split':(2, inputs['min samples split'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=RFR_TT, pbounds=RFRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['min_samples_leaf'] = int(params_best['min_samples_leaf'])
                        params_best['min_samples_split'] = int(params_best['min_samples_split'])
                        st.write("\n","\n","best params: ", params_best)

                        reg.model = RFR(n_estimators=params_best['n_estimators'],random_state=inputs['random state'],max_depth=params_best['max_depth'],min_samples_leaf=params_best['min_samples_leaf'],
                                    min_samples_split=params_best['min_samples_split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
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
                    if inputs['auto hyperparameters'] == False:
                        reg.model = SVR(kernel=inputs['kernel'], C=inputs['C'])
                        
                        reg.SupportVector()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        
                        plot_and_export_results(reg, "SVR")
                    
                    elif inputs['auto hyperparameters']:
                        def SVR_TT(C):                            
                            reg.model = SVR(kernel='rbf', C=C)
                            reg.SupportVector()
                            return reg.score
                        
                        SVRbounds = {'C':(0.001, inputs['C'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=SVR_TT, pbounds=SVRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['kernel'] = 'rbf'
                        st.write("\n","\n","best params: ", params_best)
                        
                        reg.model = SVR(kernel='rbf', C=params_best['C'])

                        reg.SupportVector()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        plot_and_export_results(reg, "SVR")                    
                elif operator == 'cross val score':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = SVR(kernel=inputs['kernel'], C=inputs['C'])

                        export_cross_val_results(reg, cv, "SVR_cv",inputs['random state'])
                    elif inputs['auto hyperparameters']:
                        def SVR_TT(C):                            
                            reg.model = SVR(kernel='rbf', C=C)
                            cv_score = cv_cal(reg, cv, inputs['random state'])
                            return cv_score
                        
                        SVRbounds = {'C':(0.001, inputs['C'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=SVR_TT, pbounds=SVRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['kernel'] = 'rbf'
                        st.write("\n","\n","best params: ", params_best)
                        
                        reg.model = SVR(kernel='rbf', C=params_best['C'])

                        export_cross_val_results(reg, cv, "SVR_cv",inputs['random state']) 

                elif operator == 'leave one out':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = SVR(kernel=inputs['kernel'], C=inputs['C'])
                    
                        export_loo_results(reg, loo, "SVR_loo")
                    elif inputs['auto hyperparameters']:
                        def SVR_TT(C):                            
                            reg.model = SVR(kernel='rbf', C=C)
                            loo_score = loo_cal(reg, loo)
                            return loo_score
                        
                        SVRbounds = {'C':(0.001, inputs['C'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=SVR_TT, pbounds=SVRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['kernel'] = 'rbf'
                        st.write("\n","\n","best params: ", params_best)
                        
                        reg.model = SVR(kernel='rbf', C=params_best['C'])

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
                    elif inputs['kernel'] == 'DotProduct+WhiteKernel':
                        kernel = DotProduct() + WhiteKernel()
                    elif inputs['kernel'] == 'Matern':
                        kernel = Matern()
                    elif inputs['kernel'] == 'PairwiseKernel':
                        kernel = PairwiseKernel()
                    elif inputs['kernel'] == 'RationalQuadratic':
                        kernel = RationalQuadratic()
                    elif inputs['kernel'] == 'RBF':
                        kernel = RBF()
                    elif inputs['kernel'] == 'DotProduct+RationalQuadratic':
                        kernel = DotProduct() + RationalQuadratic()
                    elif inputs['kernel'] == 'PairwiseKernel+RationalQuadratic':
                        kernel = PairwiseKernel() + RationalQuadratic()
                    elif inputs['kernel'] == 'DotProduct+PairwiseKernel':
                        kernel = DotProduct() + PairwiseKernel()


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
                    elif inputs['kernel'] == 'DotProduct+WhiteKernel':
                        kernel = DotProduct() + WhiteKernel()
                    elif inputs['kernel'] == 'Matern':
                        kernel = Matern()
                    elif inputs['kernel'] == 'PairwiseKernel':
                        kernel = PairwiseKernel()
                    elif inputs['kernel'] == 'RationalQuadratic':
                        kernel = RationalQuadratic()
                    elif inputs['kernel'] == 'RBF':
                        kernel = RBF()
                    elif inputs['kernel'] == 'DotProduct+RationalQuadratic':
                        kernel = DotProduct() + RationalQuadratic()
                    elif inputs['kernel'] == 'PairwiseKernel+RationalQuadratic':
                        kernel = PairwiseKernel() + RationalQuadratic()
                    elif inputs['kernel'] == 'DotProduct+PairwiseKernel':
                        kernel = DotProduct() + PairwiseKernel() 
                    reg.model = GPR(kernel = kernel, random_state = inputs['random state'])

                    export_cross_val_results(reg, cv, "GPR_cv",inputs['random state'])

                elif operator == 'leave one out':
                    if inputs['kernel'] == None:
                        kernel = None
                    elif inputs['kernel'] == 'DotProduct':
                        kernel = DotProduct()
                    elif inputs['kernel'] == 'WhiteKernel':
                        kernel = WhiteKernel()
                    elif inputs['kernel'] == 'DotProduct+WhiteKernel':
                        kernel = DotProduct() + WhiteKernel()
                    elif inputs['kernel'] == 'Matern':
                        kernel = Matern()
                    elif inputs['kernel'] == 'PairwiseKernel':
                        kernel = PairwiseKernel()
                    elif inputs['kernel'] == 'RationalQuadratic':
                        kernel = RationalQuadratic()
                    elif inputs['kernel'] == 'RBF':
                        kernel = RBF()
                    elif inputs['kernel'] == 'DotProduct+RationalQuadratic':
                        kernel = DotProduct() + RationalQuadratic()
                    elif inputs['kernel'] == 'PairwiseKernel+RationalQuadratic':
                        kernel = PairwiseKernel() + RationalQuadratic()
                    elif inputs['kernel'] == 'DotProduct+PairwiseKernel':
                        kernel = DotProduct() + PairwiseKernel()
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
                    if inputs['auto hyperparameters'] == False:
                        reg.model = KNeighborsRegressor(n_neighbors = inputs['n neighbors'])
                        
                        reg.KNeighborsRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        
                        plot_and_export_results(reg, "KNR")
                    elif inputs['auto hyperparameters']:
                        def KNNR_TT(n_neighbors):                            
                            reg.model = KNeighborsRegressor(n_neighbors = int(n_neighbors))
                            reg.KNeighborsRegressor()
                            return reg.score
                        
                        KNNRbounds = {'n_neighbors':(1, inputs['n neighbors'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=KNNR_TT, pbounds=KNNRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_neighbors'] = int(params_best['n_neighbors'])
                        st.write("\n","\n","best params: ", params_best)
                        
                        reg.model = KNeighborsRegressor(n_neighbors = params_best['n_neighbors'])

                        reg.KNeighborsRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        plot_and_export_results(reg, "KNNR")   
                elif operator == 'cross val score':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = KNeighborsRegressor(n_neighbors = inputs['n neighbors'])

                        export_cross_val_results(reg, cv, "KNR_cv", inputs['random state'])
                    elif inputs['auto hyperparameters']:
                        def KNNR_TT(n_neighbors):                            
                            reg.model = KNeighborsRegressor(n_neighbors = int(n_neighbors))
                            cv_score = cv_cal(reg, cv, inputs['random state'])
                            return cv_score
                        
                        KNNRbounds = {'n_neighbors':(1, inputs['n neighbors'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=KNNR_TT, pbounds=KNNRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_neighbors'] = int(params_best['n_neighbors'])
                        st.write("\n","\n","best params: ", params_best)
                        
                        reg.model = KNeighborsRegressor(n_neighbors = params_best['n_neighbors'])
                        export_cross_val_results(reg, cv, "KNR_cv", inputs['random state'])

                elif operator == 'leave one out':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = KNeighborsRegressor(n_neighbors = inputs['n neighbors'])
                    
                        export_loo_results(reg, loo, "KNR_loo")
                    elif inputs['auto hyperparameters']:
                        def KNNR_TT(n_neighbors):                            
                            reg.model = KNeighborsRegressor(n_neighbors = int(n_neighbors))
                            loo_score = loo_cal(reg, loo)
                            return loo_score
                        
                        KNNRbounds = {'n_neighbors':(1, inputs['n neighbors'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=KNNR_TT, pbounds=KNNRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_neighbors'] = int(params_best['n_neighbors'])
                        st.write("\n","\n","best params: ", params_best)
                        
                        reg.model = KNeighborsRegressor(n_neighbors = params_best['n_neighbors'])
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
                    if inputs['auto hyperparameters'] == False:
                        reg.model = Lasso(alpha=inputs['alpha'],random_state=inputs['random state'])
                        
                        reg.LassoRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        
                        plot_and_export_results(reg, "LassoR")
                    
                    elif inputs['auto hyperparameters']:
                        def LassoR_TT(alpha):                            
                            reg.model = Lasso(alpha=alpha)
                            reg.LassoRegressor()
                            return reg.score
                        
                        LassoRbounds = {'alpha':(0.001, inputs['alpha'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=LassoR_TT, pbounds=LassoRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        st.write("\n","\n","best params: ", params_best)
                        
                        reg.model = Lasso(alpha=params_best['alpha'],random_state=inputs['random state'])

                        reg.LassoRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        plot_and_export_results(reg, "LassoR")   
                elif operator == 'cross val score':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = Lasso(alpha=inputs['alpha'],random_state=inputs['random state'])

                        export_cross_val_results(reg, cv, "LassoR_cv", inputs['random state'])
                                                
                    elif inputs['auto hyperparameters']:
                        def LassoR_TT(alpha):                            
                            reg.model = Lasso(alpha=alpha)
                            cv_score = cv_cal(reg, cv, inputs['random state'])
                            return cv_score
                        
                        LassoRbounds = {'alpha':(0.001, inputs['alpha'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=LassoR_TT, pbounds=LassoRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        st.write("\n","\n","best params: ", params_best)
                        
                        reg.model = Lasso(alpha=params_best['alpha'],random_state=inputs['random state'])
                        export_cross_val_results(reg, cv, "LassoR_cv", inputs['random state'])

                elif operator == 'leave one out':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = Lasso(alpha=inputs['alpha'],random_state=inputs['random state'])
                    
                        export_loo_results(reg, loo, "LassoR_loo")
                    elif inputs['auto hyperparameters']:
                        def LassoR_TT(alpha):                            
                            reg.model = Lasso(alpha=alpha)
                            loo_score = loo_cal(reg, loo)
                            return loo_score
                        
                        LassoRbounds = {'alpha':(0.001, inputs['alpha'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=LassoR_TT, pbounds=LassoRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        st.write("\n","\n","best params: ", params_best)
                        
                        reg.model = Lasso(alpha=params_best['alpha'],random_state=inputs['random state'])
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
                    if inputs['auto hyperparameters'] == False:
                        reg.model = Ridge(alpha=inputs['alpha'], random_state=inputs['random state'])
                        
                        reg.RidgeRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']

                        plot_and_export_results(reg, "RidgeR")
                    
                    elif inputs['auto hyperparameters']:
                        def RidgeR_TT(alpha):                            
                            reg.model = Ridge(alpha=alpha)
                            reg.RidgeRegressor()
                            return reg.score
                        
                        RidgeRbounds = {'alpha':(0.001, inputs['alpha'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=RidgeR_TT, pbounds=RidgeRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        st.write("\n","\n","best params: ", params_best)
                        
                        reg.model = Ridge(alpha=params_best['alpha'], random_state=inputs['random state'])

                        reg.RidgeRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        plot_and_export_results(reg, "RidgeR")   
                elif operator == 'cross val score':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = Ridge(alpha=inputs['alpha'], random_state=inputs['random state'])

                        export_cross_val_results(reg, cv, "RidgeR_cv", inputs['random state'])
                    elif inputs['auto hyperparameters']:
                        def RidgeR_TT(alpha):                            
                            reg.model = Ridge(alpha=alpha)
                            cv_score = cv_cal(reg, cv, inputs['random state'])
                            return cv_score
                        
                        RidgeRbounds = {'alpha':(0.001, inputs['alpha'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=RidgeR_TT, pbounds=RidgeRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        st.write("\n","\n","best params: ", params_best)
                        
                        reg.model = Ridge(alpha=params_best['alpha'], random_state=inputs['random state'])
                        export_cross_val_results(reg, cv, "RidgeR_cv", inputs['random state'])

                elif operator == 'leave one out':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = Ridge(alpha=inputs['alpha'], random_state=inputs['random state'])
                    
                        export_loo_results(reg, loo, "RidgeR_loo")
                    elif inputs['auto hyperparameters']:
                        def RidgeR_TT(alpha):                            
                            reg.model = Ridge(alpha=alpha)
                            loo_score = loo_cal(reg, loo)
                            return loo_score
                        
                        RidgeRbounds = {'alpha':(0.001, inputs['alpha'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=RidgeR_TT, pbounds=RidgeRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        st.write("\n","\n","best params: ", params_best)
                        
                        reg.model = Ridge(alpha=params_best['alpha'], random_state=inputs['random state'])
                        export_loo_results(reg, loo, "RidgeR_loo")

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
                    if inputs['auto hyperparameters'] == False:
                        reg.model = GradientBoostingRegressor(learning_rate=inputs['learning rate'],n_estimators=inputs['nestimators'],max_features=inputs['max features'],
                                                            random_state=inputs['random state']) 
                        
                        reg.GradientBoostingRegressor()
        
                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        plot_and_export_results(reg, "GradientBoostingR")

                    elif inputs['auto hyperparameters']:
                        def GBR_TT(learning_rate, n_estimators):                            
                            reg.model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=int(n_estimators),max_features=inputs['max features'])                                           
                            reg.GradientBoostingRegressor()
                            return reg.score
                        
                        GBRbounds = {'learning_rate':(0.001, inputs['learning rate']), 'n_estimators':(1, inputs['nestimators'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=GBR_TT, pbounds=GBRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_features'] = inputs['max features']
                        st.write("\n","\n","best params: ", params_best)
                        reg.model = GradientBoostingRegressor(learning_rate=params_best['learning_rate'],n_estimators=params_best['n_estimators'],max_features=params_best['max_features'],
                                                            random_state=inputs['random state']) 
                        reg.GradientBoostingRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        plot_and_export_results(reg, "GBR")   
                elif operator == 'cross val score':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = GradientBoostingRegressor(learning_rate=inputs['learning rate'],n_estimators=inputs['nestimators'],max_features=inputs['max features'],
                                                            random_state=inputs['random state'])  

                        export_cross_val_results(reg, cv, "GradientBoostingR_cv", inputs['random state'])
                    elif inputs['auto hyperparameters']:
                        def GBR_TT(learning_rate, n_estimators):                            
                            reg.model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=int(n_estimators),max_features=inputs['max features'])                                           
                            cv_score = cv_cal(reg, cv, inputs['random state'])
                            return cv_score         
                        
                        GBRbounds = {'learning_rate':(0.001, inputs['learning rate']), 'n_estimators':(1, inputs['nestimators'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=GBR_TT, pbounds=GBRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_features'] = inputs['max features']
                        st.write("\n","\n","best params: ", params_best)
                        reg.model = GradientBoostingRegressor(learning_rate=params_best['learning_rate'],n_estimators=params_best['n_estimators'],max_features=params_best['max_features'],
                                                            random_state=inputs['random state']) 
                        export_cross_val_results(reg, cv, "GradientBoostingR_cv", inputs['random state'])

                elif operator == 'leave one out':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = GradientBoostingRegressor(learning_rate=inputs['learning rate'],n_estimators=inputs['nestimators'],max_features=inputs['max features'],
                                                            random_state=inputs['random state']) 
                        export_loo_results(reg, loo, "GradientBoostingR_loo")
                    elif inputs['auto hyperparameters']:
                        def GBR_TT(learning_rate, n_estimators):                            
                            reg.model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=int(n_estimators),max_features=inputs['max features'])                                           
                            loo_score = loo_cal(reg, loo)
                            return loo_score       
                        
                        GBRbounds = {'learning_rate':(0.001, inputs['learning rate']), 'n_estimators':(1, inputs['nestimators'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=GBR_TT, pbounds=GBRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_features'] = inputs['max features']
                        st.write("\n","\n","best params: ", params_best)
                        reg.model = GradientBoostingRegressor(learning_rate=params_best['learning_rate'],n_estimators=params_best['n_estimators'],max_features=params_best['max_features'],
                                                            random_state=inputs['random state']) 
                        export_loo_results(reg, loo, "GradientBoostingR_loo")                 

        if inputs['model'] == 'XGBRegressor':
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
                        reg.features = pd.DataFrame(reg.features)    
                        loo = LeaveOneOut()

            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train: 
                if operator == 'train test split':
                    if inputs['base estimator'] == "gbtree": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'] , n_estimators=inputs['nestimators'], 
                                                        max_depth= inputs['max depth'], subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                        learning_rate=inputs['learning rate'], random_state = inputs['random state'])
                            reg.XGBRegressor()
            
                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual','prediction']
                            
                            plot_and_export_results(reg, "XGBR")

                        elif inputs['auto hyperparameters']:
                            def XGBR_TT(n_estimators, max_depth, subsample, colsample_bytree, learning_rate):
                                
                                reg.model = xgb.XGBRegressor(booster=inputs['base estimator'] , n_estimators=int(n_estimators), 
                                                            max_depth= int(max_depth), subsample=subsample, colsample_bytree=colsample_bytree, 
                                                            learning_rate=learning_rate)
                                reg.XGBRegressor()
                                return reg.score
                            
                            XGBRbounds = {'n_estimators':(1, inputs['nestimators']), 'max_depth':(1, inputs['max depth']), 'subsample':(0.5, inputs['subsample']), 'colsample_bytree':(0.5, inputs['subsample']), 'learning_rate':(0.001, inputs['learning rate'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=XGBR_TT, pbounds=XGBRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_depth'] = int(params_best['max_depth'])
                            params_best['base estimator'] = 'gbtree'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'] , n_estimators=params_best['n_estimators'], 
                                                        max_depth= params_best['max_depth'], subsample=params_best['subsample'], colsample_bytree=params_best['colsample_bytree'], 
                                                        learning_rate=params_best['learning_rate'])
                            
                            reg.XGBRegressor()

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual','prediction']
                            plot_and_export_results(reg, "XGBR")

                    elif inputs['base estimator'] == "gblinear": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'], n_estimators=inputs['nestimators'], 
                                                        max_depth= inputs['max depth'], subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                        learning_rate=inputs['learning rate'], random_state = inputs['random state'])
                            reg.XGBRegressor()

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual','prediction']

                            plot_and_export_results(reg, "XGBR")

                        elif inputs['auto hyperparameters']:
                            def XGBR_TT(n_estimators, max_depth, subsample, colsample_bytree, learning_rate):
                                
                                reg.model = xgb.XGBRegressor(booster=inputs['base estimator'] , n_estimators=int(n_estimators), 
                                                            max_depth= int(max_depth), subsample=subsample, colsample_bytree=colsample_bytree, 
                                                            learning_rate=learning_rate)
                                reg.XGBRegressor()
                                return reg.score
                
                            XGBRbounds = {'n_estimators':(1, inputs['nestimators']), 'max_depth':(1, inputs['max depth']), 'subsample':(0.5, inputs['subsample']), 'colsample_bytree':(0.5, inputs['subsample']), 'learning_rate':(0.001, inputs['learning rate'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=XGBR_TT, pbounds=XGBRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_depth'] = int(params_best['max_depth'])
                            params_best['base estimator'] = 'gblinear'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'] , n_estimators=params_best['n_estimators'], 
                                                        max_depth= params_best['max_depth'], subsample=params_best['subsample'], colsample_bytree=params_best['colsample_bytree'], 
                                                        learning_rate=params_best['learning_rate'])
                            
                            reg.XGBRegressor()

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual','prediction']
                            plot_and_export_results(reg, "XGBR")

                elif operator == 'cross val score':
                    if inputs['base estimator'] == "gbtree": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'] , n_estimators=inputs['nestimators'], 
                                                        max_depth= inputs['max depth'], subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                        learning_rate=inputs['learning rate'], random_state = inputs['random state'])

                            cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)

                            export_cross_val_results(reg, cv, "XGBR_cv", inputs['random state'])
                        elif inputs['auto hyperparameters']:
                            def XGBR_TT(n_estimators, max_depth, subsample, colsample_bytree, learning_rate):
                                
                                reg.model = xgb.XGBRegressor(booster=inputs['base estimator'] , n_estimators=int(n_estimators), 
                                                            max_depth= int(max_depth), subsample=subsample, colsample_bytree=colsample_bytree, 
                                                            learning_rate=learning_rate)
                                cv_score = cv_cal(reg, cv, inputs['random state'])
                                return cv_score 
                            
                            XGBRbounds = {'n_estimators':(1, inputs['nestimators']), 'max_depth':(1, inputs['max depth']), 'subsample':(0.5, inputs['subsample']), 'colsample_bytree':(0.5, inputs['subsample']), 'learning_rate':(0.001, inputs['learning rate'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=XGBR_TT, pbounds=XGBRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_depth'] = int(params_best['max_depth'])
                            params_best['base estimator'] = 'gbtree'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'] , n_estimators=params_best['n_estimators'], 
                                                        max_depth= params_best['max_depth'], subsample=params_best['subsample'], colsample_bytree=params_best['colsample_bytree'], 
                                                        learning_rate=params_best['learning_rate'])
                            
                            export_cross_val_results(reg, cv, "XGBR_cv", inputs['random state'])      

                    elif inputs['base estimator'] == "gblinear": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'] , n_estimators=inputs['nestimators'], 
                                                        max_depth= inputs['max depth'], subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                        learning_rate=inputs['learning rate'], random_state = inputs['random state'])
                            # cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)
                
                            export_cross_val_results(reg, cv, "XGBR_cv", inputs['random state'])
                        elif inputs['auto hyperparameters']:
                            def XGBR_TT(n_estimators, max_depth, subsample, colsample_bytree, learning_rate):
                                
                                reg.model = xgb.XGBRegressor(booster=inputs['base estimator'] , n_estimators=int(n_estimators), 
                                                            max_depth= int(max_depth), subsample=subsample, colsample_bytree=colsample_bytree, 
                                                            learning_rate=learning_rate)
                                cv_score = cv_cal(reg, cv, inputs['random state'])
                                return cv_score 
                            
                            XGBRbounds = {'n_estimators':(1, inputs['nestimators']), 'max_depth':(1, inputs['max depth']), 'subsample':(0.5, inputs['subsample']), 'colsample_bytree':(0.5, inputs['subsample']), 'learning_rate':(0.001, inputs['learning rate'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=XGBR_TT, pbounds=XGBRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_depth'] = int(params_best['max_depth'])
                            params_best['base estimator'] = 'gbtree'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'] , n_estimators=params_best['n_estimators'], 
                                                        max_depth= params_best['max_depth'], subsample=params_best['subsample'], colsample_bytree=params_best['colsample_bytree'], 
                                                        learning_rate=params_best['learning_rate'])
                            
                            export_cross_val_results(reg, cv, "XGBR_cv", inputs['random state'])                                 

                elif operator == 'leave one out':
                    if inputs['base estimator'] == "gbtree": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'], n_estimators=inputs['nestimators'], 
                                                        max_depth= inputs['max depth'], subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                        learning_rate=inputs['learning rate'], random_state = inputs['random state'])  
                            export_loo_results(reg, loo, "XGBR_loo")
                        elif inputs['auto hyperparameters']:
                            def XGBR_TT(n_estimators, max_depth, subsample, colsample_bytree, learning_rate):
                                
                                reg.model = xgb.XGBRegressor(booster=inputs['base estimator'] , n_estimators=int(n_estimators), 
                                                            max_depth= int(max_depth), subsample=subsample, colsample_bytree=colsample_bytree, 
                                                            learning_rate=learning_rate)
                                loo_score = loo_cal(reg, loo)
                                return loo_score       
                            
                            XGBRbounds = {'n_estimators':(1, inputs['nestimators']), 'max_depth':(1, inputs['max depth']), 'subsample':(0.5, inputs['subsample']), 'colsample_bytree':(0.5, inputs['subsample']), 'learning_rate':(0.001, inputs['learning rate'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=XGBR_TT, pbounds=XGBRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_depth'] = int(params_best['max_depth'])
                            params_best['base estimator'] = 'gbtree'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'] , n_estimators=params_best['n_estimators'], 
                                                        max_depth= params_best['max_depth'], subsample=params_best['subsample'], colsample_bytree=params_best['colsample_bytree'], 
                                                        learning_rate=params_best['learning_rate'])
                            
                            export_loo_results(reg, loo, "XGBR_loo")                            

                    elif inputs['base estimator'] == "gblinear": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'], n_estimators=inputs['nestimators'], 
                                                        max_depth= inputs['max depth'], subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                        learning_rate=inputs['learning rate'], random_state = inputs['random state'])  
                            
                            export_loo_results(reg, loo, "XGBR_loo")
                        
                        elif inputs['auto hyperparameters']:
                            def XGBR_TT(n_estimators, max_depth, subsample, colsample_bytree, learning_rate):
                                
                                reg.model = xgb.XGBRegressor(booster=inputs['base estimator'] , n_estimators=int(n_estimators), 
                                                            max_depth= int(max_depth), subsample=subsample, colsample_bytree=colsample_bytree, 
                                                            learning_rate=learning_rate)
                                loo_score = loo_cal(reg, loo)
                                return loo_score       
                            
                            XGBRbounds = {'n_estimators':(1, inputs['nestimators']), 'max_depth':(1, inputs['max depth']), 'subsample':(0.5, inputs['subsample']), 'colsample_bytree':(0.5, inputs['subsample']), 'learning_rate':(0.001, inputs['learning rate'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=XGBR_TT, pbounds=XGBRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_depth'] = int(params_best['max_depth'])
                            params_best['base estimator'] = 'gbtree'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'] , n_estimators=params_best['n_estimators'], 
                                                        max_depth= params_best['max_depth'], subsample=params_best['subsample'], colsample_bytree=params_best['colsample_bytree'], 
                                                        learning_rate=params_best['learning_rate'])
                            
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
                    if inputs['auto hyperparameters'] == False:
                        reg.model =CatBoostRegressor(iterations=inputs['niteration'],learning_rate=inputs['learning rate'],depth = inputs['max depth'], random_seed=inputs['random state'])

                        reg.CatBRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']

                        plot_and_export_results(reg, "CatBoostR")
                    elif inputs['auto hyperparameters']:
                        def CatBR_TT(iterations, depth, learning_rate): 
                            reg.model = CatBoostRegressor(iterations=int(iterations),learning_rate=learning_rate,depth = int(depth))
                            reg.CatBRegressor()
                            return reg.score
                        
                        CatBRbounds = {'iterations':(1, inputs['niteration']), 'depth':(1, inputs['max depth']), 'learning_rate':(0.001, inputs['learning rate'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=CatBR_TT, pbounds=CatBRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['iterations'] = int(params_best['iterations'])
                        params_best['depth'] = int(params_best['depth'])
                        st.write("\n","\n","best params: ", params_best)
                        
                        reg.model = CatBoostRegressor(iterations=params_best['iterations'],learning_rate=params_best['learning_rate'],depth = params_best['depth'], random_seed=inputs['random state'])
                        
                        reg.CatBRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        plot_and_export_results(reg, "CatBR")
                elif operator == 'cross val score':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = CatBoostRegressor(iterations=inputs['niteration'],learning_rate=inputs['learning rate'],depth = inputs['max depth'], random_seed=inputs['random state'])
                        
                        export_cross_val_results(reg, cv, "CatBoostR_cv", inputs['random state'])
                    elif inputs['auto hyperparameters']:
                        def CatBR_TT(iterations, depth, learning_rate): 
                            reg.model = CatBoostRegressor(iterations=int(iterations),learning_rate=learning_rate,depth = int(depth))
                            cv_score = cv_cal(reg, cv, inputs['random state'])
                            return cv_score 
                        
                        CatBRbounds = {'iterations':(1, inputs['niteration']), 'depth':(1, inputs['max depth']), 'learning_rate':(0.001, inputs['learning rate'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=CatBR_TT, pbounds=CatBRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['iterations'] = int(params_best['iterations'])
                        params_best['depth'] = int(params_best['depth'])
                        st.write("\n","\n","best params: ", params_best)
                        
                        reg.model = CatBoostRegressor(iterations=params_best['iterations'],learning_rate=params_best['learning_rate'],depth = params_best['depth'], random_seed=inputs['random state'])
                        export_cross_val_results(reg, cv, "CatBoostR_cv", inputs['random state'])

                elif operator == 'leave one out':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = CatBoostRegressor(iterations=inputs['niteration'],learning_rate=inputs['learning rate'],depth = inputs['max depth'], random_seed=inputs['random state'])
                        export_loo_results(reg, loo, "CatBoostR_loo")            
                    elif inputs['auto hyperparameters']:
                        def CatBR_TT(iterations, depth, learning_rate): 
                            reg.model = CatBoostRegressor(iterations=int(iterations),learning_rate=learning_rate,depth = int(depth))
                            loo_score = loo_cal(reg, loo)
                            return loo_score       
                        
                        CatBRbounds = {'iterations':(1, inputs['niteration']), 'depth':(1, inputs['max depth']), 'learning_rate':(0.001, inputs['learning rate'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=CatBR_TT, pbounds=CatBRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['iterations'] = int(params_best['iterations'])
                        params_best['depth'] = int(params_best['depth'])
                        st.write("\n","\n","best params: ", params_best)
                        
                        reg.model = CatBoostRegressor(iterations=params_best['iterations'],learning_rate=params_best['learning_rate'],depth = params_best['depth'], random_seed=inputs['random state'])
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
                    if inputs['auto hyperparameters'] == False:
                        reg.model = MLPRegressor(hidden_layer_sizes=inputs['hidden layer size'], activation= inputs['activation'], solver=inputs['solver'], 
                                                batch_size=inputs['batch size'], learning_rate= inputs['learning rate'], max_iter=inputs['max iter'],
                                                random_state=inputs['random state'])
                        reg.MLPRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        plot_and_export_results(reg, "MLP")
                    elif inputs['auto hyperparameters']:
                        def MLPR_TT(layer_size, neuron_size): 
                            layer_size = int(layer_size)
                            neuron_size = int(neuron_size)
                            hidden_layer_size = tuple([neuron_size]*layer_size)
                            reg.model = MLPRegressor(hidden_layer_sizes=hidden_layer_size, activation= inputs['activation'], solver=inputs['solver'], 
                                                batch_size=inputs['batch size'], learning_rate=inputs['learning rate'], max_iter=inputs['max iter'],
                                                random_state=inputs['random state'])
                            reg.MLPRegressor()
                            return reg.score
                        
                        MLPRbounds = {'layer_size':(1, inputs['layer size']), 'neuron_size':(1, inputs['neuron size'])}

                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=MLPR_TT, pbounds=MLPRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['layer_size'] = int(params_best['layer_size'])
                        params_best['neuron_size'] = int(params_best['neuron_size'])
                        st.write("\n","\n","best params: ", params_best)
                        hidden_layer_size = tuple(params_best['layer_size']*[params_best['neuron_size']])
                        reg.model = MLPRegressor(hidden_layer_sizes=hidden_layer_size, activation= inputs['activation'], solver=inputs['solver'], 
                                                batch_size=inputs['batch size'], learning_rate=inputs['learning rate'], max_iter=inputs['max iter'],
                                                random_state=inputs['random state'])
                        
                        reg.MLPRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        plot_and_export_results(reg, "MLPR")                        

                elif operator == 'cross val score':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = MLPRegressor(hidden_layer_sizes=inputs['hidden layer size'], activation= inputs['activation'], solver=inputs['solver'], 
                                                batch_size=inputs['batch size'], learning_rate= inputs['learning rate'], max_iter=inputs['max iter'],
                                                random_state=inputs['random state'])
                        
                        export_cross_val_results(reg, cv, "MLP_cv", inputs['random state'])
                    elif inputs['auto hyperparameters']:
                        def MLPR_TT(layer_size, neuron_size): 
                            layer_size = int(layer_size)
                            neuron_size = int(neuron_size)
                            hidden_layer_size = tuple([neuron_size]*layer_size)
                            reg.model = MLPRegressor(hidden_layer_sizes=hidden_layer_size, activation= inputs['activation'], solver=inputs['solver'], 
                                                batch_size=inputs['batch size'], learning_rate=inputs['learning rate'], max_iter=inputs['max iter'],
                                                random_state=inputs['random state'])
                            cv_score = cv_cal(reg, cv, inputs['random state'])
                            return cv_score 
                        
                        MLPRbounds = {'layer_size':(1, inputs['layer size']), 'neuron_size':(1, inputs['neuron size'])}

                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=MLPR_TT, pbounds=MLPRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['layer_size'] = int(params_best['layer_size'])
                        params_best['neuron_size'] = int(params_best['neuron_size'])
                        st.write("\n","\n","best params: ", params_best)
                        hidden_layer_size = tuple(params_best['layer_size']*[params_best['neuron_size']])
                        reg.model = MLPRegressor(hidden_layer_sizes=hidden_layer_size, activation= inputs['activation'], solver=inputs['solver'], 
                                                batch_size=inputs['batch size'], learning_rate=inputs['learning rate'], max_iter=inputs['max iter'],
                                                random_state=inputs['random state'])
                        
                        export_cross_val_results(reg, cv, "MLP_cv", inputs['random state'])
                elif operator == 'leave one out':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = MLPRegressor(hidden_layer_sizes=inputs['hidden layer size'], activation= inputs['activation'], solver=inputs['solver'], 
                                                batch_size=inputs['batch size'], learning_rate= inputs['learning rate'], max_iter=inputs['max iter'],
                                                random_state=inputs['random state'])
                    
                        export_loo_results(reg, loo, "MLP_loo")
                    elif inputs['auto hyperparameters']:
                        def MLPR_TT(layer_size, neuron_size): 
                            layer_size = int(layer_size)
                            neuron_size = int(neuron_size)
                            hidden_layer_size = tuple([neuron_size]*layer_size)
                            reg.model = MLPRegressor(hidden_layer_sizes=hidden_layer_size, activation= inputs['activation'], solver=inputs['solver'], 
                                                batch_size=inputs['batch size'], learning_rate=inputs['learning rate'], max_iter=inputs['max iter'],
                                                random_state=inputs['random state'])
                            loo_score = loo_cal(reg, loo)
                            return loo_score     
                        
                        MLPRbounds = {'layer_size':(1, inputs['layer size']), 'neuron_size':(1, inputs['neuron size'])}

                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=MLPR_TT, pbounds=MLPRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['layer_size'] = int(params_best['layer_size'])
                        params_best['neuron_size'] = int(params_best['neuron_size'])
                        st.write("\n","\n","best params: ", params_best)
                        hidden_layer_size = tuple(params_best['layer_size']*[params_best['neuron_size']])
                        reg.model = MLPRegressor(hidden_layer_sizes=hidden_layer_size, activation= inputs['activation'], solver=inputs['solver'], 
                                                batch_size=inputs['batch size'], learning_rate=inputs['learning rate'], max_iter=inputs['max iter'],
                                                random_state=inputs['random state'])
                        
                        export_loo_results(reg, loo, "MLP_loo")                     
            st.write('---')

        if inputs['model'] == 'BaggingRegressor':
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
                        reg.features = pd.DataFrame(reg.features)    
                        loo = LeaveOneOut()

            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:

                if operator == 'train test split':
                    if inputs['base estimator'] == "DecisionTree": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = BaggingRegressor(estimator = None ,n_estimators=inputs['nestimators'],
                                    max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
                            
                            reg.BaggingRegressor()

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual','prediction']

                            plot_and_export_results(reg, "BaggingR")
                        elif inputs['auto hyperparameters']:
                            def BaggingR_TT(n_estimators, max_samples, max_features):
                                
                                reg.model = BaggingRegressor(estimator = None ,n_estimators=int(n_estimators),
                                    max_samples=int(max_samples), max_features=int(max_features), n_jobs=-1) 
                                reg.BaggingRegressor()
                                return reg.score
                            
                            BaggingRbounds = {'n_estimators':(1, inputs['nestimators']), 'max_samples':(1, inputs['max samples']), 'max_features':(1, inputs['max features'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=BaggingR_TT, pbounds=BaggingRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_samples'] = int(params_best['max_samples'])
                            params_best['max_features'] = int(params_best['max_features'])
                            params_best['base estimator'] = 'decision tree'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = BaggingRegressor(estimator = None ,n_estimators=params_best['n_estimators'],
                                    max_samples=params_best['max_samples'], max_features=params_best['max_features'], n_jobs=-1) 
                            
                            reg.BaggingRegressor()

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual','prediction']
                            plot_and_export_results(reg, "BaggingR")

                    elif inputs['base estimator'] == "SupportVector": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = BaggingRegressor(estimator = SVR(),n_estimators=inputs['nestimators'],
                                    max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
                            reg.BaggingRegressor()

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual','prediction']

                            plot_and_export_results(reg, "BaggingR")
                        elif inputs['auto hyperparameters']:
                            def BaggingR_TT(n_estimators, max_samples, max_features):
                                
                                reg.model = BaggingRegressor(estimator = SVR() ,n_estimators=int(n_estimators),
                                    max_samples=int(max_samples), max_features=int(max_features), n_jobs=-1) 
                                reg.BaggingRegressor()
                                return reg.score
                            
                            BaggingRbounds = {'n_estimators':(1, inputs['nestimators']), 'max_samples':(1, inputs['max samples']), 'max_features':(1, inputs['max features'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=BaggingR_TT, pbounds=BaggingRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_samples'] = int(params_best['max_samples'])
                            params_best['max_features'] = int(params_best['max_features'])
                            params_best['base estimator'] = 'support vector machine'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = BaggingRegressor(estimator = SVR() ,n_estimators=params_best['n_estimators'],
                                    max_samples=params_best['max_samples'], max_features=params_best['max_features'], n_jobs=-1) 
                            
                            reg.BaggingRegressor()

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual','prediction']
                            plot_and_export_results(reg, "BaggingR")

                    elif inputs['base estimator'] == "LinearRegression": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = BaggingRegressor(estimator = LinearR(),n_estimators=inputs['nestimators'],
                                    max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
                            reg.BaggingRegressor()
            

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual','prediction']

                            plot_and_export_results(reg, "BaggingR")
                        elif inputs['auto hyperparameters']:
                            def BaggingR_TT(n_estimators, max_samples, max_features):
                                
                                reg.model = BaggingRegressor(estimator = LinearR() ,n_estimators=int(n_estimators),
                                    max_samples=int(max_samples), max_features=int(max_features), n_jobs=-1) 
                                reg.BaggingRegressor()
                                return reg.score
                            
                            BaggingRbounds = {'n_estimators':(1, inputs['nestimators']), 'max_samples':(1, inputs['max samples']), 'max_features':(1, inputs['max features'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=BaggingR_TT, pbounds=BaggingRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_samples'] = int(params_best['max_samples'])
                            params_best['max_features'] = int(params_best['max_features'])
                            params_best['base estimator'] = 'linear regression'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = BaggingRegressor(estimator = LinearR() ,n_estimators=params_best['n_estimators'],
                                    max_samples=params_best['max_samples'], max_features=params_best['max_features'], n_jobs=-1) 
                            
                            reg.BaggingRegressor()

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual','prediction']
                            plot_and_export_results(reg, "BaggingR")

                elif operator == 'cross val score':
                    if inputs['base estimator'] == "DecisionTree": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = BaggingRegressor(estimator = None, n_estimators=inputs['nestimators'],
                                                            max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1)
                            # cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)

                            export_cross_val_results(reg, cv, "BaggingR_cv", inputs['random state'])
                        elif inputs['auto hyperparameters']:
                            def BaggingR_TT(n_estimators, max_samples, max_features):
                                
                                reg.model = BaggingRegressor(estimator = None ,n_estimators=int(n_estimators),
                                    max_samples=int(max_samples), max_features=int(max_features), n_jobs=-1) 
                                cv_score = cv_cal(reg, cv, inputs['random state'])
                                return cv_score 
                            
                            BaggingRbounds = {'n_estimators':(1, inputs['nestimators']), 'max_samples':(1, inputs['max samples']), 'max_features':(1, inputs['max features'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=BaggingR_TT, pbounds=BaggingRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_samples'] = int(params_best['max_samples'])
                            params_best['max_features'] = int(params_best['max_features'])
                            params_best['base estimator'] = 'decision tree'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = BaggingRegressor(estimator = None,n_estimators=params_best['n_estimators'],
                                    max_samples=params_best['max_samples'], max_features=params_best['max_features'], n_jobs=-1) 
                            
                            export_cross_val_results(reg, cv, "BaggingR_cv", inputs['random state'])

                    elif inputs['base estimator'] == "SupportVector": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = BaggingRegressor(estimator =  SVR(),n_estimators=inputs['nestimators'],
                                    max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 

                            # cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)
                
                            export_cross_val_results(reg, cv, "BaggingR_cv", inputs['random state'])
                        
                        elif inputs['auto hyperparameters']:
                            def BaggingR_TT(n_estimators, max_samples, max_features):
                                
                                reg.model = BaggingRegressor(estimator = SVR(), n_estimators=int(n_estimators),
                                    max_samples=int(max_samples), max_features=int(max_features), n_jobs=-1) 
                                cv_score = cv_cal(reg, cv, inputs['random state'])
                                return cv_score 
                            
                            BaggingRbounds = {'n_estimators':(1, inputs['nestimators']), 'max_samples':(1, inputs['max samples']), 'max_features':(1, inputs['max features'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=BaggingR_TT, pbounds=BaggingRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_samples'] = int(params_best['max_samples'])
                            params_best['max_features'] = int(params_best['max_features'])
                            params_best['base estimator'] = 'support vector machine'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = BaggingRegressor(estimator = None, n_estimators=params_best['n_estimators'],
                                    max_samples=params_best['max_samples'], max_features=params_best['max_features'], n_jobs=-1) 
                            
                            export_cross_val_results(reg, cv, "BaggingR_cv", inputs['random state'])                        

                    elif inputs['base estimator'] == "LinearRegression":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = BaggingRegressor(estimator = LinearR(),n_estimators=inputs['nestimators'],
                                    max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 

                            # cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)
                
                            export_cross_val_results(reg, cv, "BaggingR_cv", inputs['random state'])
                        elif inputs['auto hyperparameters']:
                            def BaggingR_TT(n_estimators, max_samples, max_features):
                                reg.model = BaggingRegressor(estimator = LinearR(), n_estimators=int(n_estimators),
                                    max_samples=int(max_samples), max_features=int(max_features), n_jobs=-1) 
                                cv_score = cv_cal(reg, cv, inputs['random state'])
                                return cv_score 
                            
                            BaggingRbounds = {'n_estimators':(1, inputs['nestimators']), 'max_samples':(1, inputs['max samples']), 'max_features':(1, inputs['max features'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=BaggingR_TT, pbounds=BaggingRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_samples'] = int(params_best['max_samples'])
                            params_best['max_features'] = int(params_best['max_features'])
                            params_best['base estimator'] = 'linear regression'
                            st.write("\n","\n","best params: ", params_best)               
                            reg.model = BaggingRegressor(estimator = LinearR(), n_estimators=params_best['n_estimators'],
                                    max_samples=params_best['max_samples'], max_features=params_best['max_features'], n_jobs=-1) 
                            
                            export_cross_val_results(reg, cv, "BaggingR_cv", inputs['random state'])    
                elif operator == 'leave one out':
                    if inputs['base estimator'] == "DecisionTree": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = BaggingRegressor(estimator = tree.DecisionTreeRegressor(random_state=inputs['random state']),n_estimators=inputs['nestimators'],
                                    max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
                        
                            export_loo_results(reg, loo, "BaggingR_loo")
                        elif inputs['auto hyperparameters']:

                            def BaggingR_TT(n_estimators, max_samples, max_features):
                                reg.model = BaggingRegressor(estimator = None, n_estimators=int(n_estimators),
                                    max_samples=int(max_samples), max_features=int(max_features), n_jobs=-1) 
                                loo_score = loo_cal(reg, loo)
                                return loo_score     
                            
                            BaggingRbounds = {'n_estimators':(1, inputs['nestimators']), 'max_samples':(1, inputs['max samples']), 'max_features':(1, inputs['max features'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=BaggingR_TT, pbounds=BaggingRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_samples'] = int(params_best['max_samples'])
                            params_best['max_features'] = int(params_best['max_features'])
                            params_best['base estimator'] = 'decision tree'
                            st.write("\n","\n","best params: ", params_best)               
                            reg.model = BaggingRegressor(estimator = None, n_estimators=params_best['n_estimators'],
                                    max_samples=params_best['max_samples'], max_features=params_best['max_features'], n_jobs=-1) 
                            export_loo_results(reg, loo, "BaggingR_loo") 
                    
                    elif inputs['base estimator'] == "SupportVector": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = BaggingRegressor(estimator = SVR(),n_estimators=inputs['nestimators'],
                                    max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
                            export_loo_results(reg, loo, "BaggingR_loo")
                        elif inputs['auto hyperparameters']:
                            def BaggingR_TT(n_estimators, max_samples, max_features):
                                reg.model = BaggingRegressor(estimator = SVR(), n_estimators=int(n_estimators),
                                    max_samples=int(max_samples), max_features=int(max_features), n_jobs=-1) 
                                loo_score = loo_cal(reg, loo)
                                return loo_score     
                            
                            BaggingRbounds = {'n_estimators':(1, inputs['nestimators']), 'max_samples':(1, inputs['max samples']), 'max_features':(1, inputs['max features'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=BaggingR_TT, pbounds=BaggingRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_samples'] = int(params_best['max_samples'])
                            params_best['max_features'] = int(params_best['max_features'])
                            params_best['base estimator'] = 'support vector machine'
                            st.write("\n","\n","best params: ", params_best)               
                            reg.model = BaggingRegressor(estimator = SVR(), n_estimators=params_best['n_estimators'],
                                    max_samples=params_best['max_samples'], max_features=params_best['max_features'], n_jobs=-1) 
                            export_loo_results(reg, loo, "BaggingR_loo")  

                    elif inputs['base estimator'] == "LinearRegression":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = BaggingRegressor(estimator = LinearR(),n_estimators=inputs['nestimators'],
                                    max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
            
                            export_loo_results(reg, loo, "BaggingR_loo")
                        elif inputs['auto hyperparameters']:
                            def BaggingR_TT(n_estimators, max_samples, max_features):
                                reg.model = BaggingRegressor(estimator = LinearR(), n_estimators=int(n_estimators),
                                    max_samples=int(max_samples), max_features=int(max_features), n_jobs=-1) 
                                loo_score = loo_cal(reg, loo)
                                return loo_score     
                            
                            BaggingRbounds = {'n_estimators':(1, inputs['nestimators']), 'max_samples':(1, inputs['max samples']), 'max_features':(1, inputs['max features'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=BaggingR_TT, pbounds=BaggingRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_samples'] = int(params_best['max_samples'])
                            params_best['max_features'] = int(params_best['max_features'])
                            params_best['base estimator'] = 'support vector machine'
                            st.write("\n","\n","best params: ", params_best)               
                            reg.model = BaggingRegressor(estimator = LinearR(), n_estimators=params_best['n_estimators'],
                                    max_samples=params_best['max_samples'], max_features=params_best['max_features'], n_jobs=-1) 
                            export_loo_results(reg, loo, "BaggingR_loo")                              

        if inputs['model'] == 'AdaBoostRegressor':
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
                        reg.features = pd.DataFrame(reg.features)    
                        loo = LeaveOneOut()

            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:

                if operator == 'train test split':

                    if inputs['base estimator'] == "DecisionTree": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state'])                             
                            reg.AdaBoostRegressor()
                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual','prediction']
                            plot_and_export_results(reg, "AdaBoostR")

                        elif inputs['auto hyperparameters']:
                            def AdaBoostR_TT(n_estimators, learning_rate):
                                reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(), 
                                                            n_estimators=int(n_estimators), learning_rate=learning_rate) 
                                reg.AdaBoostRegressor()
                                return reg.score
                            
                            AdaBoostRbounds = {'n_estimators':(1, inputs['nestimators']), 'learning_rate':(1, inputs['learning rate'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=AdaBoostR_TT, pbounds=AdaBoostRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['base estimator'] = 'decision tree'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(), 
                                                            n_estimators=params_best['n_estimators'], learning_rate=params_best['learning_rate'], random_state=inputs['random state'])                
                            reg.AdaBoostRegressor()
                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual','prediction']
                            plot_and_export_results(reg, "AdaBoostR")                        
                    
                    elif inputs['base estimator'] == "SupportVector": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = AdaBoostRegressor(estimator=SVR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 
                            reg.AdaBoostRegressor()
                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual','prediction']
                            plot_and_export_results(reg, "AdaBoostR")
                        elif inputs['auto hyperparameters']:
                            def AdaBoostR_TT(n_estimators, learning_rate):
                                reg.model = AdaBoostRegressor(estimator=SVR(), 
                                                            n_estimators=int(n_estimators), learning_rate=learning_rate) 
                                reg.AdaBoostRegressor()
                                return reg.score
                            
                            AdaBoostRbounds = {'n_estimators':(1, inputs['nestimators']), 'learning_rate':(1, inputs['learning rate'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=AdaBoostR_TT, pbounds=AdaBoostRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['base estimator'] = 'support vector machine'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = AdaBoostRegressor(estimator=SVR(), 
                                                            n_estimators=params_best['n_estimators'], learning_rate=params_best['learning_rate'], random_state=inputs['random state'])                
                            reg.AdaBoostRegressor()
                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual','prediction']
                            plot_and_export_results(reg, "AdaBoostR")                                 


                    elif inputs['base estimator'] == "LinearRegression": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = AdaBoostRegressor(estimator=LinearR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state'])
                            reg.AdaBoostRegressor()
                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual','prediction']
                            plot_and_export_results(reg, "AdaBoostR")
                        elif inputs['auto hyperparameters']:
                            def AdaBoostR_TT(n_estimators, learning_rate):
                                reg.model = AdaBoostRegressor(estimator=LinearR(), 
                                                            n_estimators=int(n_estimators), learning_rate=learning_rate) 
                                reg.AdaBoostRegressor()
                                return reg.score
                            
                            AdaBoostRbounds = {'n_estimators':(1, inputs['nestimators']), 'learning_rate':(1, inputs['learning rate'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=AdaBoostR_TT, pbounds=AdaBoostRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['base estimator'] = 'linear regression'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = AdaBoostRegressor(estimator=LinearR(), 
                                                            n_estimators=params_best['n_estimators'], learning_rate=params_best['learning_rate'], random_state=inputs['random state'])                
                            reg.AdaBoostRegressor()
                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual','prediction']
                            plot_and_export_results(reg, "AdaBoostR")                                    

                elif operator == 'cross val score':
                    if inputs['base estimator'] == "DecisionTree": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 
                            # cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)
                            export_cross_val_results(reg, cv, "AdaBoostR_cv", inputs['random state'])
                        elif inputs['auto hyperparameters']:
                            def AdaBoostR_TT(n_estimators, learning_rate):
                                reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(), 
                                                            n_estimators=int(n_estimators), learning_rate=learning_rate) 
                                cv_score = cv_cal(reg, cv, inputs['random state'])
                                return cv_score 
                            
                            AdaBoostRbounds = {'n_estimators':(1, inputs['nestimators']), 'learning_rate':(1, inputs['learning rate'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=AdaBoostR_TT, pbounds=AdaBoostRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['base estimator'] = 'decision tree'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(), 
                                                            n_estimators=params_best['n_estimators'], learning_rate=params_best['learning_rate'], random_state=inputs['random state'])                
                            
                            export_cross_val_results(reg, cv, "AdaBoostR_cv", inputs['random state'])


                    elif inputs['base estimator'] == "SupportVector": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = AdaBoostRegressor(estimator=SVR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 

                            # cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)
                
                            export_cross_val_results(reg, cv, "AdaBoostR_cv", inputs['random state'])
                        elif inputs['auto hyperparameters']:
                            def AdaBoostR_TT(n_estimators, learning_rate):
                                reg.model = AdaBoostRegressor(estimator=SVR(), 
                                                            n_estimators=int(n_estimators), learning_rate=learning_rate) 
                                cv_score = cv_cal(reg, cv, inputs['random state'])
                                return cv_score 
                            
                            AdaBoostRbounds = {'n_estimators':(1, inputs['nestimators']), 'learning_rate':(1, inputs['learning rate'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=AdaBoostR_TT, pbounds=AdaBoostRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['base estimator'] = 'support vector machine'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = AdaBoostRegressor(estimator=SVR(), 
                                                            n_estimators=params_best['n_estimators'], learning_rate=params_best['learning_rate'], random_state=inputs['random state'])                
                            
                            export_cross_val_results(reg, cv, "AdaBoostR_cv", inputs['random state'])

                    elif inputs['base estimator'] == "LinearRegression":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = reg.model = AdaBoostRegressor(estimator=LinearR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state'])
                            # cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)
                
                            export_cross_val_results(reg, cv, "AdaBoostR_cv", inputs['random state'])
                        elif inputs['auto hyperparameters']:
                            def AdaBoostR_TT(n_estimators, learning_rate):
                                reg.model = AdaBoostRegressor(estimator=LinearR(), 
                                                            n_estimators=int(n_estimators), learning_rate=learning_rate) 
                                cv_score = cv_cal(reg, cv, inputs['random state'])
                                return cv_score 
                            
                            AdaBoostRbounds = {'n_estimators':(1, inputs['nestimators']), 'learning_rate':(1, inputs['learning rate'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=AdaBoostR_TT, pbounds=AdaBoostRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['base estimator'] = 'linear regression'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = AdaBoostRegressor(estimator=LinearR(), 
                                                            n_estimators=params_best['n_estimators'], learning_rate=params_best['learning_rate'], random_state=inputs['random state'])                
                            
                            export_cross_val_results(reg, cv, "AdaBoostR_cv", inputs['random state'])
                elif operator == 'leave one out':
                    if inputs['base estimator'] == "DecisionTree": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state'])  
                        
                            export_loo_results(reg, loo, "AdaBoostR_loo")
                        elif inputs['auto hyperparameters']:
                            def AdaBoostR_TT(n_estimators, learning_rate):
                                reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(), 
                                                            n_estimators=int(n_estimators), learning_rate=learning_rate) 
                                loo_score = loo_cal(reg, loo)
                                return loo_score     
                            
                            AdaBoostRbounds = {'n_estimators':(1, inputs['nestimators']), 'learning_rate':(1, inputs['learning rate'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=AdaBoostR_TT, pbounds=AdaBoostRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['base estimator'] = 'decision tree'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(), 
                                                            n_estimators=params_best['n_estimators'], learning_rate=params_best['learning_rate'], random_state=inputs['random state'])                
                            
                            export_loo_results(reg, loo, "AdaBoostR_loo")
                            
                    elif inputs['base estimator'] == "SupportVector": 
                        if inputs['auto hyperparameters'] == False:
                            reg.model = AdaBoostRegressor(estimator=SVR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 
                            
                            export_loo_results(reg, loo, "AdaBoostR_loo")
                        elif inputs['auto hyperparameters']:
                            def AdaBoostR_TT(n_estimators, learning_rate):
                                reg.model = AdaBoostRegressor(estimator=SVR(), 
                                                            n_estimators=int(n_estimators), learning_rate=learning_rate) 
                                loo_score = loo_cal(reg, loo)
                                return loo_score     
                            
                            AdaBoostRbounds = {'n_estimators':(1, inputs['nestimators']), 'learning_rate':(1, inputs['learning rate'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=AdaBoostR_TT, pbounds=AdaBoostRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['base estimator'] = 'support vector machine'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = AdaBoostRegressor(estimator=SVR(), 
                                                            n_estimators=params_best['n_estimators'], learning_rate=params_best['learning_rate'], random_state=inputs['random state'])                
                            
                            export_loo_results(reg, loo, "AdaBoostR_loo")

                    elif inputs['base estimator'] == "LinearRegression":
                        if inputs['auto hyperparameters'] == False:
                            reg.model =  AdaBoostRegressor(estimator=LinearR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 
            
                            export_loo_results(reg, loo, "AdaBoostR_loo")
                        elif inputs['auto hyperparameters']:
                            def AdaBoostR_TT(n_estimators, learning_rate):
                                reg.model = AdaBoostRegressor(estimator=LinearR(), 
                                                            n_estimators=int(n_estimators), learning_rate=learning_rate) 
                                loo_score = loo_cal(reg, loo)
                                return loo_score     
                            
                            AdaBoostRbounds = {'n_estimators':(1, inputs['nestimators']), 'learning_rate':(1, inputs['learning rate'])}
                            
                            with st.expander('hyperparameter opt'):
                                optimizer = BayesianOptimization(f=AdaBoostR_TT, pbounds=AdaBoostRbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['base estimator'] = 'linear regression'
                            st.write("\n","\n","best params: ", params_best)
                            
                            reg.model = AdaBoostRegressor(estimator=LinearR(), 
                                                            n_estimators=params_best['n_estimators'], learning_rate=params_best['learning_rate'], random_state=inputs['random state'])                
                            
                            export_loo_results(reg, loo, "AdaBoostR_loo")                            
    st.write('---')                
                            
elif select_option == "Classification":

    colored_header(label="Classification",description=" ",color_name="violet-90")
    file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
    if file is None:
        table = PrettyTable(['file name', 'class','description'])
        table.add_row(['file_1','dataset','data file'])
        st.write(table)
    if file is not None:
        df = pd.read_csv(file)
        check_string(df)
        colored_header(label="Data information", description=" ",color_name="violet-70")
        nrow = st.slider("rows", 1, len(df), 5)
        df_nrow = df.head(nrow)
        st.write(df_nrow)

        colored_header(label="Feature and target",description=" ",color_name="violet-70")

        target_num = st.number_input('target number',  min_value=1, max_value=10, value=1)
        
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

        col_name = list(clf.targets)
        clf.targets[col_name[0]], unique_categories = pd.factorize(clf.targets[col_name[0]])

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
                    if inputs['auto hyperparameters'] == False:
                        clf.model = tree.DecisionTreeClassifier(criterion=inputs['criterion'],random_state=inputs['random state'],splitter=inputs['splitter'],
                                                                max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],min_samples_split=inputs['min samples split'])

                        clf.DecisionTreeClassifier()
                        plot_and_export_results_clf(clf, 'DTC', col_name, unique_categories)
                        if inputs['tree graph']:
                            class_names = list(clf.features)
                            dot_data = tree.export_graphviz(clf.model,out_file=None, feature_names=list(clf.features), class_names=class_names,filled=True, rounded=True)
                            graph = graphviz.Source(dot_data)
                            graph.render('Tree graph', view=True)
                    
                    elif inputs['auto hyperparameters']:
                        def DTC_TT(max_depth, min_samples_leaf, min_samples_split):
                            clf.model = tree.DecisionTreeClassifier(criterion=inputs['criterion'],random_state=inputs['random state'],splitter=inputs['splitter'],
                                                                max_depth=int(max_depth),min_samples_leaf=int(min_samples_leaf),min_samples_split=int(min_samples_split))
                            clf.DecisionTreeClassifier()
                            return clf.score
                        
                        DTCbounds = {'max_depth':(1, inputs['max depth']), 'min_samples_leaf':(1, inputs['min samples leaf']), 'min_samples_split':(2, inputs['min samples split'])}
                        
                        with st.expander('hyperparameter opt'):
                        
                            optimizer = BayesianOptimization(f=DTC_TT, pbounds=DTCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['min_samples_leaf'] = int(params_best['min_samples_leaf'])
                        params_best['min_samples_split'] = int(params_best['min_samples_split'])
                        st.write("\n","\n","best params: ", params_best)                        
                        clf.model = tree.DecisionTreeClassifier(criterion=inputs['criterion'],random_state=inputs['random state'],splitter=inputs['splitter'],
                                        max_depth=params_best['max_depth'],min_samples_leaf=params_best['min_samples_leaf'],
                                        min_samples_split=params_best['min_samples_split'])                     
                        clf.DecisionTreeClassifier()
                        plot_and_export_results_clf(clf, 'DTC', col_name, unique_categories)
                        if inputs['tree graph']:
                            class_names = list(clf.targets)
                            dot_data = tree.export_graphviz(clf.model,out_file=None, feature_names=list(clf.features), class_names=class_names,filled=True, rounded=True)
                            graph = graphviz.Source(dot_data)
                            graph.render('Tree graph', view=True)                        

                elif data_process == 'cross val score':
                    if inputs['auto hyperparameters'] == False:
                        clf.model = tree.DecisionTreeClassifier(criterion=inputs['criterion'],random_state=inputs['random state'],splitter=inputs['splitter'],
                                                                max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],min_samples_split=inputs['min samples split'])
                                                                
                        export_cross_val_results_clf(clf, cv, "DTC_cv", col_name, unique_categories, inputs['random state'])
                        if inputs['tree graph']:
                            class_names = list(clf.targets)
                            dot_data = tree.export_graphviz(clf.model,out_file=None, feature_names=list(clf.features), class_names=class_names,filled=True, rounded=True)
                            graph = graphviz.Source(dot_data)
                            graph.render('Tree graph', view=True)    
                    elif inputs['auto hyperparameters']:
                        def DTC_TT(max_depth, min_samples_leaf, min_samples_split):
                            clf.model = tree.DecisionTreeClassifier(criterion=inputs['criterion'],random_state=inputs['random state'],splitter=inputs['splitter'],
                                                                max_depth=int(max_depth),min_samples_leaf=int(min_samples_leaf),min_samples_split=int(min_samples_split))
                            cv_score = cv_cal_clf(clf, cv, inputs['random state'])
                            return cv_score
                        
                        DTCbounds = {'max_depth':(1, inputs['max depth']), 'min_samples_leaf':(1, inputs['min samples leaf']), 'min_samples_split':(2, inputs['min samples split'])}
                        
                        with st.expander('hyperparameter opt'):
                        
                            optimizer = BayesianOptimization(f=DTC_TT, pbounds=DTCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['min_samples_leaf'] = int(params_best['min_samples_leaf'])
                        params_best['min_samples_split'] = int(params_best['min_samples_split'])
                        st.write("\n","\n","best params: ", params_best)                        
                        clf.model = tree.DecisionTreeClassifier(criterion=inputs['criterion'],random_state=inputs['random state'],splitter=inputs['splitter'],
                                        max_depth=params_best['max_depth'],min_samples_leaf=params_best['min_samples_leaf'],
                                        min_samples_split=params_best['min_samples_split'])                     
                        export_cross_val_results_clf(clf, cv, "DTC_cv", col_name, unique_categories, inputs['random state'])
                        if inputs['tree graph']:
                            class_names = list(clf.targets)
                            dot_data = tree.export_graphviz(clf.model,out_file=None, feature_names=list(clf.features), class_names=class_names,filled=True, rounded=True)
                            graph = graphviz.Source(dot_data)
                            graph.render('Tree graph', view=True)                   
                elif data_process == 'leave one out':
                    if inputs['auto hyperparameters'] == False:
                        clf.model = tree.DecisionTreeClassifier(criterion=inputs['criterion'],random_state=inputs['random state'],splitter=inputs['splitter'],
                                                                max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],min_samples_split=inputs['min samples split'])                        
                        export_loo_results_clf(clf, loo, "DTC_loo", col_name, unique_categories)
                    elif inputs['auto hyperparameters']:
                        def DTC_TT(max_depth, min_samples_leaf, min_samples_split):
                            clf.model = tree.DecisionTreeClassifier(criterion=inputs['criterion'],random_state=inputs['random state'],splitter=inputs['splitter'],
                                                                max_depth=int(max_depth),min_samples_leaf=int(min_samples_leaf),min_samples_split=int(min_samples_split))
                            loo_score = loo_cal_clf(clf, loo)
                            return loo_score
                        
                        DTCbounds = {'max_depth':(1, inputs['max depth']), 'min_samples_leaf':(1, inputs['min samples leaf']), 'min_samples_split':(2, inputs['min samples split'])}
                        
                        with st.expander('hyperparameter opt'):
                        
                            optimizer = BayesianOptimization(f=DTC_TT, pbounds=DTCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['min_samples_leaf'] = int(params_best['min_samples_leaf'])
                        params_best['min_samples_split'] = int(params_best['min_samples_split'])
                        st.write("\n","\n","best params: ", params_best)                        
                        clf.model = tree.DecisionTreeClassifier(criterion=inputs['criterion'],random_state=inputs['random state'],splitter=inputs['splitter'],
                                        max_depth=params_best['max_depth'],min_samples_leaf=params_best['min_samples_leaf'],
                                        min_samples_split=params_best['min_samples_split'])                     
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
                    if inputs['auto hyperparameters'] == False:
                        clf.model = RFC(criterion = inputs['criterion'],n_estimators=inputs['nestimators'],random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                    min_samples_split=inputs['min samples split'], warm_start=inputs['warm start'])
                        
                        clf.RandomForestClassifier()
                        plot_and_export_results_clf(clf, 'RFC', col_name, unique_categories)
                    elif inputs['auto hyperparameters']:
                        def RFC_TT(n_estimators, max_depth, min_samples_leaf, min_samples_split):
                            clf.model = RFC(criterion = inputs['criterion'],n_estimators=int(n_estimators),max_depth=int(max_depth),min_samples_leaf=int(min_samples_leaf),
                                                        min_samples_split=int(min_samples_split), n_jobs=-1)
                            clf.RandomForestClassifier()
                            return clf.score
                        
                        RFCbounds = {'n_estimators':(1, inputs['nestimators']), 'max_depth':(1, inputs['max depth']), 'min_samples_leaf':(1, inputs['min samples leaf']), 'min_samples_split':(2, inputs['min samples split'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=RFC_TT, pbounds=RFCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['min_samples_leaf'] = int(params_best['min_samples_leaf'])
                        params_best['min_samples_split'] = int(params_best['min_samples_split'])
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = RFC(criterion = inputs['criterion'], n_estimators=params_best['n_estimators'],random_state=inputs['random state'],max_depth=params_best['max_depth'],min_samples_leaf=params_best['min_samples_leaf'],
                                    min_samples_split=params_best['min_samples_split'],warm_start=inputs['warm start'],
                                    n_jobs=-1)
                        
                        clf.RandomForestClassifier()
                        plot_and_export_results_clf(clf, 'RFC', col_name, unique_categories)

                elif data_process == 'cross val score':
                    if inputs['auto hyperparameters'] == False:
                        clf.model = RFC(criterion = inputs['criterion'],n_estimators=inputs['nestimators'],random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                    min_samples_split=inputs['min samples split'], warm_start=inputs['warm start'])

                        export_cross_val_results_clf(clf, cv, "RFC_cv", col_name, unique_categories, inputs['random state'])
                    elif inputs['auto hyperparameters']:
                        def RFC_TT(n_estimators, max_depth, min_samples_leaf, min_samples_split):
                            clf.model = RFC(criterion = inputs['criterion'],n_estimators=int(n_estimators),max_depth=int(max_depth),min_samples_leaf=int(min_samples_leaf),
                                                        min_samples_split=int(min_samples_split), n_jobs=-1)
                            cv_score = cv_cal_clf(clf, cv, inputs['random state'])
                            return cv_score
                        
                        RFCbounds = {'n_estimators':(1, inputs['nestimators']), 'max_depth':(1, inputs['max depth']), 'min_samples_leaf':(1, inputs['min samples leaf']), 'min_samples_split':(2, inputs['min samples split'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=RFC_TT, pbounds=RFCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['min_samples_leaf'] = int(params_best['min_samples_leaf'])
                        params_best['min_samples_split'] = int(params_best['min_samples_split'])
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = RFC(criterion = inputs['criterion'], n_estimators=params_best['n_estimators'],random_state=inputs['random state'],max_depth=params_best['max_depth'],min_samples_leaf=params_best['min_samples_leaf'],
                                    min_samples_split=params_best['min_samples_split'],warm_start=inputs['warm start'],
                                    n_jobs=-1)
                        
                        export_cross_val_results_clf(clf, cv, "RFC_cv", col_name, unique_categories, inputs['random state'])
                elif data_process == 'leave one out':
                    if inputs['auto hyperparameters'] == False:
                        clf.model = RFC(criterion = inputs['criterion'],n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                    min_samples_split=inputs['min samples split'], warm_start=inputs['warm start'])
                    
                        export_loo_results_clf(clf, loo, "RFC_loo", col_name, unique_categories)     
                    elif inputs['auto hyperparameters']:
                        def RFC_TT(n_estimators, max_depth, min_samples_leaf, min_samples_split):
                            clf.model = RFC(criterion = inputs['criterion'],n_estimators=int(n_estimators),max_depth=int(max_depth),min_samples_leaf=int(min_samples_leaf),
                                                        min_samples_split=int(min_samples_split), n_jobs=-1)
                            loo_score = loo_cal_clf(clf, loo)
                            return loo_score
                        
                        RFCbounds = {'n_estimators':(1, inputs['nestimators']), 'max_depth':(1, inputs['max depth']), 'min_samples_leaf':(1, inputs['min samples leaf']), 'min_samples_split':(2, inputs['min samples split'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=RFC_TT, pbounds=RFCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['min_samples_leaf'] = int(params_best['min_samples_leaf'])
                        params_best['min_samples_split'] = int(params_best['min_samples_split'])
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = RFC(criterion = inputs['criterion'], n_estimators=params_best['n_estimators'],random_state=inputs['random state'],max_depth=params_best['max_depth'],min_samples_leaf=params_best['min_samples_leaf'],
                                    min_samples_split=params_best['min_samples_split'],warm_start=inputs['warm start'],
                                    n_jobs=-1)     
                        export_loo_results_clf(clf, loo, "RFC_loo", col_name, unique_categories)                         

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
                    if inputs['auto hyperparameters'] == False:
                        clf.model = LR(penalty=inputs['penalty'],C=inputs['C'],solver=inputs['solver'],max_iter=inputs['max iter'],multi_class=inputs['multi class'],
                                    random_state=inputs['random state'],l1_ratio= inputs['l1 ratio'])   
                        clf.LogisticRegreesion()
                        plot_and_export_results_clf(clf, 'LRC', col_name, unique_categories)
                    elif inputs['auto hyperparameters']:
                        def LRC_TT(C):
                            clf.model = LR(penalty=inputs['penalty'],C=C,solver=inputs['solver'],max_iter=inputs['max iter'],multi_class=inputs['multi class'],
                                        l1_ratio= inputs['l1 ratio'])   
                            clf.LogisticRegreesion()
                            return clf.score
                        LRCbounds = {'C':(1, inputs['C'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=LRC_TT, pbounds=LRCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = LR(penalty=inputs['penalty'],C=params_best['C'],solver=inputs['solver'],max_iter=inputs['max iter'],multi_class=inputs['multi class'],
                                        l1_ratio= inputs['l1 ratio'])   
                        
                        clf.LogisticRegreesion()
                        plot_and_export_results_clf(clf, 'LRC', col_name, unique_categories)                     

                elif data_process == 'cross val score':
                    if inputs['auto hyperparameters'] == False:
                        clf.model = LR(penalty=inputs['penalty'],C=inputs['C'],solver=inputs['solver'],max_iter=inputs['max iter'],multi_class=inputs['multi class'],
                                    random_state=inputs['random state'],l1_ratio= inputs['l1 ratio'])   

                        export_cross_val_results_clf(clf, cv, "LRC_cv", col_name, unique_categories, inputs['random state'])      
                    elif inputs['auto hyperparameters']:
                        def LRC_TT(C):
                            clf.model = LR(penalty=inputs['penalty'],C=C,solver=inputs['solver'],max_iter=inputs['max iter'],multi_class=inputs['multi class'],
                                        l1_ratio= inputs['l1 ratio'])   
                            cv_score = cv_cal_clf(clf, cv, inputs['random state'])
                            return cv_score
                        LRCbounds = {'C':(1, inputs['C'])}
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=LRC_TT, pbounds=LRCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = LR(penalty=inputs['penalty'],C=params_best['C'],solver=inputs['solver'],max_iter=inputs['max iter'],multi_class=inputs['multi class'],
                                        l1_ratio= inputs['l1 ratio'])   
                        
                        export_cross_val_results_clf(clf, cv, "LRC_cv", col_name, unique_categories, inputs['random state'])     

                elif data_process == 'leave one out':
                    if inputs['auto hyperparameters'] == False:
                        clf.model = LR(penalty=inputs['penalty'],C=inputs['C'],solver=inputs['solver'],max_iter=inputs['max iter'],multi_class=inputs['multi class'],
                                    random_state=inputs['random state'],l1_ratio= inputs['l1 ratio']) 
    
                        export_loo_results_clf(clf, loo, "LRC_loo", col_name, unique_categories)          
                    elif inputs['auto hyperparameters']:
                        def LRC_TT(C):
                            clf.model = LR(penalty=inputs['penalty'],C=C,solver=inputs['solver'],max_iter=inputs['max iter'],multi_class=inputs['multi class'],
                                        l1_ratio= inputs['l1 ratio'])   
                            loo_score = loo_cal_clf(clf, loo)
                            return loo_score
                        
                        LRCbounds = {'C':(1, inputs['C'])}
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=LRC_TT, pbounds=LRCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = LR(penalty=inputs['penalty'],C=params_best['C'],solver=inputs['solver'],max_iter=inputs['max iter'],multi_class=inputs['multi class'],
                                        l1_ratio= inputs['l1 ratio'])   
                        
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
                    if inputs['auto hyperparameters'] == False:        
                        clf.model = SVC(C=inputs['C'], kernel=inputs['kernel'], class_weight=inputs['class weight'])
                                                                
                        clf.SupportVector()
                        plot_and_export_results_clf(clf, 'SVC', col_name, unique_categories)                    
                    elif inputs['auto hyperparameters']:
                        def SVC_TT(C):
                            clf.model = SVC(C=C, kernel=inputs['kernel'], class_weight=inputs['class weight']) 
                            clf.SupportVector()
                            return clf.score
                        SVCbounds = {'C':(1, inputs['C'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=SVC_TT, pbounds=SVCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = SVC(C=C, kernel=inputs['kernel'], class_weight=inputs['class weight']) 
                        
                        clf.SupportVector()
                        plot_and_export_results_clf(clf, 'SVC', col_name, unique_categories)   

                elif data_process == 'cross val score':
                    if inputs['auto hyperparameters'] == False:    
                        clf.model = SVC(C=inputs['C'], kernel=inputs['kernel'], class_weight=inputs['class weight'])                                                                                       
                        export_cross_val_results_clf(clf, cv, "SVC_cv", col_name, unique_categories, inputs['random state'])     
                    elif inputs['auto hyperparameters']:
                        def SVC_TT(C):
                            clf.model = SVC(C=C, kernel=inputs['kernel'], class_weight=inputs['class weight']) 
                            cv_score = cv_cal_clf(clf, cv, inputs['random state'])
                            return cv_score
                        SVCbounds = {'C':(1, inputs['C'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=SVC_TT, pbounds=SVCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = SVC(C=C, kernel=inputs['kernel'], class_weight=inputs['class weight']) 
                        export_cross_val_results_clf(clf, cv, "SVC_cv", col_name, unique_categories, inputs['random state']) 
                        
                elif data_process == 'leave one out':
                    if inputs['auto hyperparameters'] == False:    
                        clf.model = SVC(C=inputs['C'], kernel=inputs['kernel'], class_weight=inputs['class weight'])
        
                        export_loo_results_clf(clf, loo, "SVC_loo", col_name, unique_categories)  
                    elif inputs['auto hyperparameters']:
                        def SVC_TT(C):
                            clf.model = SVC(C=C, kernel=inputs['kernel'], class_weight=inputs['class weight']) 
                            loo_score = loo_cal_clf(clf, loo)
                            return loo_score
                        SVCbounds = {'C':(1, inputs['C'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=SVC_TT, pbounds=SVCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = SVC(C=C, kernel=inputs['kernel'], class_weight=inputs['class weight']) 
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
                    if inputs['auto hyperparameters'] == False:    
                        clf.model = BaggingClassifier(estimator = tree.DecisionTreeClassifier(),n_estimators=inputs['nestimators'],
                                                    max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 

                        clf.BaggingClassifier()
                        plot_and_export_results_clf(clf, 'BaggingC', col_name, unique_categories)     
                    
                    elif inputs['auto hyperparameters']:
                        def BaggingC_TT(n_estimators, max_samples, max_features):
                            clf.model = BaggingClassifier(estimator = tree.DecisionTreeClassifier() ,n_estimators=int(n_estimators),
                                max_samples=int(max_samples), max_features=int(max_features), n_jobs=-1) 
                            clf.BaggingClassifier()
                            return clf.score
                        
                        BaggingCbounds = {'n_estimators':(1, inputs['nestimators']), 'max_samples':(1, inputs['max samples']), 'max_features':(1, inputs['max features'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=BaggingC_TT, pbounds=BaggingCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_samples'] = int(params_best['max_samples'])
                        params_best['max_features'] = int(params_best['max_features'])
                        params_best['base estimator'] = 'decision tree'
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = BaggingClassifier(estimator = tree.DecisionTreeClassifier() ,n_estimators=params_best['n_estimators'],
                                max_samples=params_best['max_samples'], max_features=params_best['max_features'], n_jobs=-1) 
                        
                        clf.BaggingClassifier()

                        plot_and_export_results_clf(clf, 'BaggingC', col_name, unique_categories)   

                elif data_process == 'cross val score':
                    if inputs['auto hyperparameters'] == False:   
                        clf.model = BaggingClassifier(estimator = tree.DecisionTreeClassifier(),n_estimators=inputs['nestimators'],
                                                    max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 

                        export_cross_val_results_clf(clf, cv, "BaggingC_cv", col_name, unique_categories, inputs['random state']) 
                    elif inputs['auto hyperparameters']:
                        def BaggingC_TT(n_estimators, max_samples, max_features):
                            clf.model = BaggingClassifier(estimator = tree.DecisionTreeClassifier() ,n_estimators=int(n_estimators),
                                max_samples=int(max_samples), max_features=int(max_features), n_jobs=-1) 
                            cv_score = cv_cal_clf(clf, cv, inputs['random state'])
                            return cv_score
                        
                        BaggingCbounds = {'n_estimators':(1, inputs['nestimators']), 'max_samples':(1, inputs['max samples']), 'max_features':(1, inputs['max features'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=BaggingC_TT, pbounds=BaggingCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_samples'] = int(params_best['max_samples'])
                        params_best['max_features'] = int(params_best['max_features'])
                        params_best['base estimator'] = 'decision tree'
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = BaggingClassifier(estimator = tree.DecisionTreeClassifier() ,n_estimators=params_best['n_estimators'],
                                max_samples=params_best['max_samples'], max_features=params_best['max_features'], n_jobs=-1) 

                        export_cross_val_results_clf(clf, cv, "BaggingC_cv", col_name, unique_categories, inputs['random state'])                    

                elif data_process == 'leave one out':
                    if inputs['auto hyperparameters'] == False:   
                        clf.model = BaggingClassifier(estimator = tree.DecisionTreeClassifier(),n_estimators=inputs['nestimators'],
                                                    max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 

                        export_loo_results_clf(clf, loo, "BaggingC_loo", col_name, unique_categories)  
                    elif inputs['auto hyperparameters']:
                        def BaggingC_TT(n_estimators, max_samples, max_features):
                            clf.model = BaggingClassifier(estimator = tree.DecisionTreeClassifier() ,n_estimators=int(n_estimators),
                                max_samples=int(max_samples), max_features=int(max_features), n_jobs=-1) 
                            loo_score = loo_cal_clf(clf, loo)
                            return loo_score
                        
                        BaggingCbounds = {'n_estimators':(1, inputs['nestimators']), 'max_samples':(1, inputs['max samples']), 'max_features':(1, inputs['max features'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=BaggingC_TT, pbounds=BaggingCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_samples'] = int(params_best['max_samples'])
                        params_best['max_features'] = int(params_best['max_features'])
                        params_best['base estimator'] = 'decision tree'
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = BaggingClassifier(estimator = tree.DecisionTreeClassifier() ,n_estimators=params_best['n_estimators'],
                                max_samples=params_best['max_samples'], max_features=params_best['max_features'], n_jobs=-1) 

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
                    if inputs['auto hyperparameters'] == False:   
                        clf.model = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state'])

                        clf.AdaBoostClassifier()
                        plot_and_export_results_clf(clf, 'AdaBoostC', col_name, unique_categories)  

                    elif inputs['auto hyperparameters']:
                        def AdaBoostC_TT(n_estimators, learning_rate):
                            clf.model = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(), 
                                                        n_estimators=int(n_estimators), learning_rate=learning_rate) 
                            clf.AdaBoostClassifier()
                            return clf.score
                        
                        AdaBoostCbounds = {'n_estimators':(1, inputs['nestimators']), 'learning_rate':(1, inputs['learning rate'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=AdaBoostC_TT, pbounds=AdaBoostCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['base estimator'] = 'decision tree'
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(), 
                                                        n_estimators=params_best['n_estimators'], learning_rate=params_best['learning_rate'], random_state=inputs['random state'])                
                        clf.AdaBoostClassifier()
                        plot_and_export_results_clf(clf, 'AdaBoostC', col_name, unique_categories)                               
                elif data_process == 'cross val score':
                    if inputs['auto hyperparameters'] == False:  
                        clf.model = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state'])
                        
                        export_cross_val_results_clf(clf, cv, "AdaBoostC_cv", col_name, unique_categories, inputs['random state']) 
                    elif inputs['auto hyperparameters']:
                        def AdaBoostC_TT(n_estimators, learning_rate):
                            clf.model = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(), 
                                                        n_estimators=int(n_estimators), learning_rate=learning_rate) 
                            cv_score = cv_cal_clf(clf, cv, inputs['random state'])
                            return cv_score
                        
                        AdaBoostCbounds = {'n_estimators':(1, inputs['nestimators']), 'learning_rate':(1, inputs['learning rate'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=AdaBoostC_TT, pbounds=AdaBoostCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['base estimator'] = 'decision tree'
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(), 
                                                        n_estimators=params_best['n_estimators'], learning_rate=params_best['learning_rate'], random_state=inputs['random state'])                

                        export_cross_val_results_clf(clf, cv, "AdaBoostC_cv", col_name, unique_categories, inputs['random state']) 

                elif data_process == 'leave one out':
                    if inputs['auto hyperparameters'] == False:  
                        clf.model = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state'])
                        
                        export_loo_results_clf(clf, loo, "AdaBoostC_loo", col_name, unique_categories)  
                    elif inputs['auto hyperparameters']:
                        def AdaBoostC_TT(n_estimators, learning_rate):
                            clf.model = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(), 
                                                        n_estimators=int(n_estimators), learning_rate=learning_rate) 
                            loo_score = loo_cal_clf(clf, loo)
                            return loo_score
                        
                        AdaBoostCbounds = {'n_estimators':(1, inputs['nestimators']), 'learning_rate':(1, inputs['learning rate'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=AdaBoostC_TT, pbounds=AdaBoostCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['base estimator'] = 'decision tree'
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(), 
                                                        n_estimators=params_best['n_estimators'], learning_rate=params_best['learning_rate'], random_state=inputs['random state'])                
                    
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
                    if inputs['auto hyperparameters'] == False:   
                        clf.model = GradientBoostingClassifier(learning_rate=inputs['learning rate'],n_estimators=inputs['nestimators'],max_depth=inputs['max depth'],max_features=inputs['max features'],
                                                            random_state=inputs['random state'])
                        clf.GradientBoostingClassifier()

                        plot_and_export_results_clf(clf, 'GBC', col_name, unique_categories)  
                    elif inputs['auto hyperparameters']:
                        def GBC_TT(learning_rate, n_estimators):                            
                            clf.model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=int(n_estimators),max_features=inputs['max features'])                                           
                            clf.GradientBoostingClassifier()
                            return clf.score
                        
                        GBCbounds = {'learning_rate':(0.001, inputs['learning rate']), 'n_estimators':(1, inputs['nestimators'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=GBC_TT, pbounds=GBCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_features'] = inputs['max features']
                        st.write("\n","\n","best params: ", params_best)
                        clf.model = GradientBoostingClassifier(learning_rate=params_best['learning_rate'],n_estimators=params_best['n_estimators'],max_features=params_best['max_features'],
                                                            random_state=inputs['random state']) 
                        clf.GradientBoostingClassifier()
                        plot_and_export_results_clf(clf, 'GBC', col_name, unique_categories)  

                elif data_process == 'cross val score':
                    if inputs['auto hyperparameters'] == False:  
                        clf.model = GradientBoostingClassifier(learning_rate=inputs['learning rate'],n_estimators=inputs['nestimators'],max_depth=inputs['max depth'],max_features=inputs['max features'],
                                                            random_state=inputs['random state'])

                        export_cross_val_results_clf(clf, cv, "GBC_cv", col_name, unique_categories, inputs['random state'])   
                    elif inputs['auto hyperparameters']:
                        def GBC_TT(learning_rate, n_estimators):                            
                            clf.model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=int(n_estimators),max_features=inputs['max features'])                                           
                            cv_score = cv_cal_clf(clf, cv, inputs['random state'])
                            return cv_score
                        GBCbounds = {'learning_rate':(0.001, inputs['learning rate']), 'n_estimators':(1, inputs['nestimators'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=GBC_TT, pbounds=GBCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_features'] = inputs['max features']
                        st.write("\n","\n","best params: ", params_best)
                        clf.model = GradientBoostingClassifier(learning_rate=params_best['learning_rate'],n_estimators=params_best['n_estimators'],max_features=params_best['max_features'],
                                                            random_state=inputs['random state']) 
                
                        export_cross_val_results_clf(clf, cv, "GBC_cv", col_name, unique_categories, inputs['random state'])   


                elif data_process == 'leave one out':
                    if inputs['auto hyperparameters'] == False:  
                        clf.model = GradientBoostingClassifier(learning_rate=inputs['learning rate'],n_estimators=inputs['nestimators'],max_depth=inputs['max depth'],max_features=inputs['max features'],
                                                            random_state=inputs['random state'])
                        export_loo_results_clf(clf, loo, "GBC_loo", col_name, unique_categories)
                    elif inputs['auto hyperparameters']:
                        def GBC_TT(learning_rate, n_estimators):                            
                            clf.model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=int(n_estimators),max_features=inputs['max features'])                                           
                            loo_score = loo_cal_clf(clf, loo)
                            return loo_score
                        
                        GBCbounds = {'learning_rate':(0.001, inputs['learning rate']), 'n_estimators':(1, inputs['nestimators'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=GBC_TT, pbounds=GBCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_features'] = inputs['max features']
                        st.write("\n","\n","best params: ", params_best)
                        clf.model = GradientBoostingClassifier(learning_rate=params_best['learning_rate'],n_estimators=params_best['n_estimators'],max_features=params_best['max_features'],
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
                    if inputs['auto hyperparameters'] == False:  
                        clf.model = xgb.XGBClassifier(booster=inputs['base estimator'], n_estimators=inputs['nestimators'], max_depth=inputs['max depth'],
                                                    subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                    learning_rate=inputs['learning rate'])
                        # clf.Ytest = clf.Ytest.reset_index(drop=True)
                        clf.XGBClassifier()
                        plot_and_export_results_clf(clf, 'XGBC', col_name, unique_categories)
                    elif inputs['auto hyperparameters']:  
                        def XGBC_TT(n_estimators, max_depth, subsample, colsample_bytree, learning_rate):
                            
                            clf.model = xgb.XGBClassifier(booster=inputs['base estimator'] , n_estimators=int(n_estimators), 
                                                        max_depth= int(max_depth), subsample=subsample, colsample_bytree=colsample_bytree, 
                                                        learning_rate=learning_rate)
                            clf.XGBClassifier()
                            return clf.score
                        
                        XGBCbounds = {'n_estimators':(1, inputs['nestimators']), 'max_depth':(1, inputs['max depth']), 'subsample':(0.5, inputs['subsample']), 'colsample_bytree':(0.5, inputs['subsample']), 'learning_rate':(0.001, inputs['learning rate'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=XGBC_TT, pbounds=XGBCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['base estimator'] = 'gbtree'
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = xgb.XGBClassifier(booster=inputs['base estimator'] , n_estimators=params_best['n_estimators'], 
                                                    max_depth= params_best['max_depth'], subsample=params_best['subsample'], colsample_bytree=params_best['colsample_bytree'], 
                                                    learning_rate=params_best['learning_rate'])
                        clf.XGBClassifier()
                        plot_and_export_results_clf(clf, 'XGBC', col_name, unique_categories)

                elif data_process == 'cross val score':
                    if inputs['auto hyperparameters'] == False:  
                        clf.model = xgb.XGBClassifier(booster=inputs['base estimator'], n_estimators=inputs['nestimators'], 
                                                    subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                    learning_rate=inputs['learning rate'])
                        

                        export_cross_val_results_clf(clf, cv, "XGBC_cv", col_name, unique_categories, inputs['random state'])  
                    elif inputs['auto hyperparameters']:  
                        def XGBC_TT(n_estimators, max_depth, subsample, colsample_bytree, learning_rate):
                            
                            clf.model = xgb.XGBClassifier(booster=inputs['base estimator'] , n_estimators=int(n_estimators), 
                                                        max_depth= int(max_depth), subsample=subsample, colsample_bytree=colsample_bytree, 
                                                        learning_rate=learning_rate)
                            cv_score = cv_cal_clf(clf, cv, inputs['random state'])
                            return cv_score
                        
                        XGBCbounds = {'n_estimators':(1, inputs['nestimators']), 'max_depth':(1, inputs['max depth']), 'subsample':(0.5, inputs['subsample']), 'colsample_bytree':(0.5, inputs['subsample']), 'learning_rate':(0.001, inputs['learning rate'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=XGBC_TT, pbounds=XGBCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['base estimator'] = 'gbtree'
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = xgb.XGBClassifier(booster=inputs['base estimator'] , n_estimators=params_best['n_estimators'], 
                                                    max_depth= params_best['max_depth'], subsample=params_best['subsample'], colsample_bytree=params_best['colsample_bytree'], 
                                                    learning_rate=params_best['learning_rate'])
                  
                        export_cross_val_results_clf(clf, cv, "XGBC_cv", col_name, unique_categories, inputs['random state'])  

                elif data_process == 'leave one out':
                    if inputs['auto hyperparameters'] == False:      
                        clf.model = xgb.XGBClassifier(booster=inputs['base estimator'], n_estimators=inputs['nestimators'], 
                                                    subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                    learning_rate=inputs['learning rate'])

                        export_loo_results_clf(clf, loo, "XGBC_loo", col_name, unique_categories)
                    elif inputs['auto hyperparameters']:  
                        def XGBC_TT(n_estimators, max_depth, subsample, colsample_bytree, learning_rate):
                            
                            clf.model = xgb.XGBClassifier(booster=inputs['base estimator'] , n_estimators=int(n_estimators), 
                                                        max_depth= int(max_depth), subsample=subsample, colsample_bytree=colsample_bytree, 
                                                        learning_rate=learning_rate)
                            loo_score = loo_cal_clf(clf, loo)
                            return loo_score
                        
                        XGBCbounds = {'n_estimators':(1, inputs['nestimators']), 'max_depth':(1, inputs['max depth']), 'subsample':(0.5, inputs['subsample']), 'colsample_bytree':(0.5, inputs['subsample']), 'learning_rate':(0.001, inputs['learning rate'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=XGBC_TT, pbounds=XGBCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['base estimator'] = 'gbtree'
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = xgb.XGBClassifier(booster=inputs['base estimator'] , n_estimators=params_best['n_estimators'], 
                                                    max_depth= params_best['max_depth'], subsample=params_best['subsample'], colsample_bytree=params_best['colsample_bytree'], 
                                                    learning_rate=params_best['learning_rate'])
                
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
                    if inputs['auto hyperparameters'] == False:    
                        clf.model = CatBoostClassifier(iterations=inputs['niteration'],learning_rate=inputs['learning rate'],depth = inputs['max depth'], random_seed=inputs['random state'])

                        clf.CatBoostClassifier()
                        plot_and_export_results_clf(clf, 'CatBoostC', col_name, unique_categories)   
                    elif inputs['auto hyperparameters']:
                        def CatBC_TT(iterations, depth, learning_rate): 
                            clf.model = CatBoostClassifier(iterations=int(iterations),learning_rate=learning_rate,depth = int(depth))
                            clf.CatBoostClassifier()
                            return clf.score
                        
                        CatBCbounds = {'iterations':(1, inputs['niteration']), 'depth':(1, inputs['max depth']), 'learning_rate':(0.001, inputs['learning rate'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=CatBC_TT, pbounds=CatBCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['iterations'] = int(params_best['iterations'])
                        params_best['depth'] = int(params_best['depth'])
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = CatBoostClassifier(iterations=params_best['iterations'],learning_rate=params_best['learning_rate'],depth = params_best['depth'], random_seed=inputs['random state'])
                        
                        clf.CatBoostClassifier()
                        plot_and_export_results_clf(clf, 'CatBoostC', col_name, unique_categories)   

                elif data_process == 'cross val score':
                    if inputs['auto hyperparameters'] == False:   
                        clf.model = CatBoostClassifier(iterations=inputs['niteration'],learning_rate=inputs['learning rate'],depth = inputs['max depth'], random_seed=inputs['random state'])

                        export_cross_val_results_clf(clf, cv, "CatBoostC_cv", col_name, unique_categories, inputs['random state'])  

                    elif inputs['auto hyperparameters']:
                        def CatBC_TT(iterations, depth, learning_rate): 
                            clf.model = CatBoostClassifier(iterations=int(iterations),learning_rate=learning_rate,depth = int(depth))
                            cv_score = cv_cal_clf(clf, cv, inputs['random state'])
                            return cv_score
                        
                        CatBCbounds = {'iterations':(1, inputs['niteration']), 'depth':(1, inputs['max depth']), 'learning_rate':(0.001, inputs['learning rate'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=CatBC_TT, pbounds=CatBCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['iterations'] = int(params_best['iterations'])
                        params_best['depth'] = int(params_best['depth'])
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = CatBoostClassifier(iterations=params_best['iterations'],learning_rate=params_best['learning_rate'],depth = params_best['depth'], random_seed=inputs['random state'])
                        export_cross_val_results_clf(clf, cv, "CatBoostC_cv", col_name, unique_categories, inputs['random state'])  

                elif data_process == 'leave one out':
                    if inputs['auto hyperparameters'] == False:      
                        clf.model = CatBoostClassifier(iterations=inputs['niteration'],learning_rate=inputs['learning rate'],depth = inputs['max depth'], random_seed=inputs['random state'])

                        export_loo_results_clf(clf, loo, "CatBoostC_loo", col_name, unique_categories)
                    elif inputs['auto hyperparameters']:
                        def CatBC_TT(iterations, depth, learning_rate): 
                            clf.model = CatBoostClassifier(iterations=int(iterations),learning_rate=learning_rate,depth = int(depth))
                            loo_score = loo_cal_clf(clf, loo)
                            return loo_score
                        
                        CatBCbounds = {'iterations':(1, inputs['niteration']), 'depth':(1, inputs['max depth']), 'learning_rate':(0.001, inputs['learning rate'])}
                        
                        with st.expander('hyperparameter opt'):
                            optimizer = BayesianOptimization(f=CatBC_TT, pbounds=CatBCbounds, random_state=inputs['random state'], allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['iterations'] = int(params_best['iterations'])
                        params_best['depth'] = int(params_best['depth'])
                        st.write("\n","\n","best params: ", params_best)
                        
                        clf.model = CatBoostClassifier(iterations=params_best['iterations'],learning_rate=params_best['learning_rate'],depth = params_best['depth'], random_seed=inputs['random state'])
                        export_loo_results_clf(clf, loo, "CatBoostC_loo", col_name, unique_categories)                 
        st.write('---')

elif select_option == "Cluster & ReduceDim":
    colored_header(label="Cluster & ReduceDim",description=" ",color_name="violet-90")
    file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
    if file is None:
        table = PrettyTable(['file name', 'class','description'])
        table.add_row(['file_1','dataset','data file'])
        st.write(table)
    if file is not None:
        df = pd.read_csv(file)
        check_string_NaN(df)
        colored_header(label="Data information", description=" ",color_name="violet-70")
        nrow = st.slider("rows", 1, len(df), 5)
        df_nrow = df.head(nrow)
        st.write(df_nrow)

        colored_header(label="Feature and target",description=" ",color_name="violet-70")

        target_num = st.number_input('target number',  min_value=1, max_value=10, value=1)
        
        col_feature, col_target = st.columns(2)
            
        # features
        features = df.iloc[:,:-target_num]
        # targets
        targets = df.iloc[:,-target_num:]
        with col_feature:    
            st.write(features.head())
        with col_target:   
            st.write(targets.head())
        cluster = CLUSTER(features,targets)

        colored_header(label="target", description=" ", color_name="violet-70")

        target_selected_name = st.selectbox('target', list(cluster.targets)[::-1])

        cluster.targets = targets[target_selected_name]
       #=============== cluster ================

        colored_header(label="Cluster & ReduceDim",description=" ",color_name="violet-70")

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
                    with st.expander('cluster'):
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
                    with st.expander('reduce dim'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'dim reduction data.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)                    
                else:
                    result_data =  PCA_transformed_data
                    with st.expander('reduce dim'):
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
                    with st.expander('reduce dim'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'dim reduction data.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)                    
                else:
                    result_data =  TSNE_transformed_data
                    with st.expander('reduce dim'):
                        st.write(result_data)                    
                        tmp_download_link = download_button(result_data, f'dim reduction data.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)           
        st.write('---')   

elif select_option == "Active Learning":
    with st.sidebar:
        sub_option = option_menu(None, ["Single-objective Active Learning", "Multi-objective Active Learning"])
    
    if sub_option == "Single-objective Active Learning":

        colored_header(label="Single-objective Active Learning",description=" ",color_name="violet-90")

        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed", accept_multiple_files=True)
        if len(file) != 2:
            table = PrettyTable(['file name', 'class','descriptor'])
            table.add_row(['file_1','dataset','data file'])
            table.add_row(['file_2','visual data','design space'])
            st.write(table)      
        if len(file) == 2:

            colored_header(label="Data information",description=" ",color_name="violet-70")

            df = pd.read_csv(file[0])
            df_vs = pd.read_csv(file[1])
            check_string_NaN(df)

            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="Feature and target",description=" ",color_name="violet-70")

            target_num = st.number_input('target number',  min_value=1, max_value=10, value=1)
            
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

            colored_header(label="target", description=" ", color_name="violet-70")

            target_selected_option = st.selectbox('target', list(sp.targets))
            
            sp.targets = sp.targets[target_selected_option]
            
            colored_header(label="Sampling", description=" ",color_name="violet-70")

            model_path = './models/active learning'

            template_alg = model_platform(model_path)

            inputs, col2 = template_alg.show()

            if inputs['model'] == 'BayeSampling':

                with col2:
                    
                    sp.vsfeatures = df_vs
                    st.info('You have upoaded the visual sample point file.')
                    feature_name = sp.features.columns.tolist()
                    if inputs['sample criterion'] == 'Augmented Expected Improvement':
                        with st.expander('EI HyperParamters'):
                            alpha = st.slider('alpha', 0.0, 3.0, 1.0)
                            tao = st.slider('tao',0.0, 1.0, 0.0)
                    if inputs['sample criterion'] == 'Expected Quantile Improvement':
                        with st.expander('EQI HyperParamters'):
                            beta= st.slider('beta',0.2, 0.8, 0.5)
                            tao = st.slider('tao_new',0.0, 1.0, 0.0)  
                    if inputs['sample criterion'] == 'Upper confidence bound':
                        with st.expander('UCB HyperParamters'):
                            alpha = st.slider('alpha', 0.0, 3.0, 1.0)   
                    if inputs['sample criterion'] == 'Probability of Improvement':
                        with st.expander('PoI HyperParamters'):
                            tao = st.slider('tao',0.0, 0.3, 0.0)  
                    if inputs['sample criterion'] == 'Predictive Entropy Search':
                        with st.expander('PES HyperParamters'):
                            sam_num = st.number_input('sample number',100, 1000, 500)     
                    if inputs['sample criterion'] == 'Knowledge Gradient':
                        with st.expander('Knowldge_G Hyperparameters'):
                            MC_num = st.number_input('MC number', 50,300,50)                      
                with st.expander('visual samples'):
                    st.write(sp.vsfeatures)
                    tmp_download_link = download_button(sp.vsfeatures, f'visual samples.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                Bgolearn = BGOS.Bgolearn()
                
                colored_header(label="Optimize", description=" ",color_name="violet-70")
                with st.container():
                    button_train = st.button('Train', use_container_width=True)
                if button_train:
                    if inputs['noise std'] != 'heteroheneous':
                        Mymodel = Bgolearn.fit(data_matrix = sp.features, Measured_response = sp.targets, virtual_samples = sp.vsfeatures,
                                            opt_num=inputs['opt num'], min_search=inputs['min search'], noise_std= float(inputs['noise std']))
                    else:
                        if 'noise' in df.columns:
                            noise_std = df['noise'].values
                            Mymodel = Bgolearn.fit(data_matrix = sp.features, Measured_response = sp.targets, virtual_samples = sp.vsfeatures,
                                                opt_num=inputs['opt num'], min_search=inputs['min search'], noise_std=noise_std)
                        else:
                            st.write("Column 'noise' is not exist")   
                        
                    if inputs['sample criterion'] == 'Expected Improvement algorith':
                        res = Mymodel.EI()
                        
                    elif inputs['sample criterion'] == 'Expected improvement with "plugin"':
                        res = Mymodel.EI_plugin()

                    elif inputs['sample criterion'] == 'Augmented Expected Improvement':
                        # with st.expander('EI HyperParamters'):
                        #     alpha = st.slider('alpha', 0.0, 3.0, 1.0)
                        #     tao = st.slider('tao',0.0, 1.0, 0.0)
                        res = Mymodel.Augmented_EI(alpha = alpha, tao = tao)

                    elif inputs['sample criterion'] == 'Expected Quantile Improvement':
                        # with st.expander('EQI HyperParamters'):
                        #     beta= st.slider('beta',0.2, 0.8, 0.5)
                        #     tao = st.slider('tao_new',0.0, 1.0, 0.0)            
                        res = Mymodel.EQI(beta = beta,tao_new = tao)

                    elif inputs['sample criterion'] == 'Reinterpolation Expected Improvement':  
                        res = Mymodel.Reinterpolation_EI() 

                    elif inputs['sample criterion'] == 'Upper confidence bound':
                        # with st.expander('UCB HyperParamters'):
                        #     alpha = st.slider('alpha', 0.0, 3.0, 1.0)
                        res = Mymodel.UCB(alpha=alpha)

                    elif inputs['sample criterion'] == 'Probability of Improvement':
                        # with st.expander('PoI HyperParamters'):
                        #     tao = st.slider('tao',0.0, 0.3, 0.0)
                        res = Mymodel.PoI(tao = tao)

                    elif inputs['sample criterion'] == 'Predictive Entropy Search':
                        # with st.expander('PES HyperParamters'):
                        #     sam_num = st.number_input('sample number',100, 1000, 500)
                        res = Mymodel.PES(sam_num = sam_num)  
                        
                    elif inputs['sample criterion'] == 'Knowledge Gradient':
                        # with st.expander('Knowldge_G Hyperparameters'):
                        #     MC_num = st.number_input('MC number', 50,300,50)
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

    elif sub_option == "Multi-objective Active Learning":

        colored_header(label="Multi-objective Active Learning",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed", accept_multiple_files=True)
        if len(file) != 2:
            table = PrettyTable(['file name', 'class','description'])
            table.add_row(['file_1','dataset','data file'])
            table.add_row(['file_2','visual data','design space'])
            st.write(table)
        elif len(file) == 2:    
            colored_header(label="Data information",description=" ",color_name="violet-70")
            # with st.expander('Data Information'):
            df = pd.read_csv(file[0])
        
            df_vs = pd.read_csv(file[1])
            check_string_NaN(df_vs)
            check_string_NaN(df)

            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)
            colored_header(label="Feature and target",description=" ",color_name="violet-70")

            target_num = st.number_input('target number',  min_value=2, max_value=2, value=2)
            
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

            colored_header(label="target", description=" ", color_name="violet-70")
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

                    vs_features = df_vs

                    reg.Xtest = vs_features
                    st.info('You have upoaded the visual sample point file.')

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
                    pareto_front = pareto_front.reset_index(drop=True)
                    st.write(pareto_front)
                    tmp_download_link = download_button(pareto_front, f'Pareto_front.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                
                ref_point = []
                for i in range(len(target_selected_option)):
                    ref_point_loc = st.number_input(target_selected_option[i] + ' ref location', 0, 10000, 0)
                    ref_point.append(ref_point_loc)
                colored_header(label="Optimize", description=" ",color_name="violet-70")
                with st.container():
                    button_train = st.button('Opt', use_container_width=True)  
                if button_train:      
                    if reg.Xtrain.columns.tolist() != reg.Xtest.columns.tolist():
                        st.error('the feature number in Visual sample file is wrong')
                        st.stop()
                    HV_value, recommend_point, Ypred_recommend = mobo.fit(X = reg.Xtrain, y = reg.Ytrain, visual_data=reg.Xtest, 
                                                    method=inputs['method'],kernel_option=inputs['kernel'],number= inputs['num'], objective=inputs['objective'], ref_point=ref_point)
                    HV_value = pd.DataFrame(HV_value, columns=["HV value"]) 

                    recommend_point = pd.DataFrame(recommend_point, columns=feature_name)  
                    
                    if inputs['normalize'] == 'StandardScaler':
                        recommend_point  = inverse_normalize(recommend_point, scaler, "StandardScaler")
                    elif inputs['normalize'] == 'MinMaxScaler':
                        recommend_point  = inverse_normalize(recommend_point, scaler, "MinMaxScaler")

                    st.write(recommend_point)
                    tmp_download_link = download_button(recommend_point, f'recommended.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True) 

                with st.expander('visual samples'):
                    if inputs['normalize'] == 'StandardScaler':
                        reg.Xtest  = inverse_normalize(reg.Xtest, scaler, "StandardScaler")
                    elif inputs['normalize'] == 'MinMaxScaler':
                        reg.Xtest  = inverse_normalize(reg.Xtest, scaler, "MinMaxScaler")
                    st.write(reg.Xtest)
                    tmp_download_link = download_button(reg.Xtest, f'visual samples.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)                   

elif select_option == "Transfer Learning":
    with st.sidebar:
        sub_option = option_menu(None, ["Boosting"])
    if sub_option == "Boosting":
        colored_header(label="Sample-based Transfer Learning ",description=" ",color_name="violet-90")

        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed", accept_multiple_files=True)
        if len(file) < 3:
            table = PrettyTable(['file name', 'class','descirption'])
            table.add_row(['file_1','test_data','target domain with no-label'])
            table.add_row(['file_2','target_data','target domain with label'])
            table.add_row(['file_3','source_data_1','1 source domain'])
            table.add_row(['...','...','...'])
            table.add_row(['file_n','source_data_n','n source domain'])
            st.write(table)

        elif len(file) >= 3:
            df_test = pd.read_csv(file[0])
            df_target = pd.read_csv(file[1])
            source_files = file[2:]
            df = [pd.read_csv(f) for f in source_files]
            df_source = pd.concat(df, axis=0)

            colored_header(label="Data information", description=" ",color_name="violet-70")

            nrow = st.slider("rows", 1, len(df_target), 5)
            df_nrow = df_target.head(nrow)
            st.write(df_nrow)
            colored_header(label="Feature and target",description=" ",color_name="violet-70")

            target_num = st.number_input('target number',  min_value=1, max_value=10, value=1)
            
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
            reg = REGRESSOR(target_features, target_targets)

            colored_header(label="target", description=" ", color_name="violet-70")

            target_selected_option = st.selectbox('target', list(reg.targets)[::-1])

            reg.targets = target_targets[target_selected_option]

            colored_header(label="Transfer", description=" ",color_name="violet-30")

            model_path = './models/transfer learning'

            template_alg = model_platform(model_path)

            inputs, col2 = template_alg.show()
 
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
                        with st.expander('prediction'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)
                    except KeyError:
                        st.write(prediction)
                        tmp_download_link = download_button(prediction, f'prediction.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)         
                    with st.expander("weak estimators"):
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

elif select_option == "Surrogate Optimization":
    with st.sidebar:
        sub_option = option_menu(None, ["Single-objective Surrogate Optimization", "Multi-objective Surrogate Optimization","Single-objective Surrogate Optimization (TL)","Multi-objective Surrogate Optimization (TL)"])
    if sub_option == "Single-objective Surrogate Optimization":

        colored_header(label="Single-objective Surrogate Optimization",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.pickle` model and `.csv` file",  label_visibility="collapsed", accept_multiple_files=True)
        if len(file) < 3:
            table = PrettyTable(['file name', 'class','description'])
            table.add_row(['file_1','dataset','data file'])
            table.add_row(['file_2','boundary','feature design constraint'])
            table.add_row(['file_3','model','model'])
            st.write(table)
        
        if len(file) >= 3:
            df = pd.read_csv(file[0])
            check_string_NaN(df)

            colored_header(label="Feature and target",description=" ",color_name="violet-70")

            target_num = st.number_input('target number',  min_value=1, max_value=10, value=1)
            
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
            colored_header(label="feature design constriant", description=" ", color_name="violet-70")
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
                        best_y= -best_y
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
                    alg = SABoltzmann(func=opt_func, x0=x0, T_max = inputs['T max'], q=inputs['q'], L=inputs['L'], max_stay_counter=inputs['max stay counter'], lb=inputs['lb'], ub=inputs['ub'])
                    best_x, best_y = alg.run()

                    loss_history = alg.best_y_history
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
                
    elif sub_option == "Multi-objective Surrogate Optimization":

        colored_header(label="Multi-objective surrogate Optimization",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.pickle` model and `.csv` file",  label_visibility="collapsed", accept_multiple_files=True)
        if len(file) < 4:
            table = PrettyTable(['file name', 'class','description'])
            table.add_row(['file_1','dataset','data file'])
            table.add_row(['file_2','boundary','design feature constraint'])
            table.add_row(['file_3','model_1','obj1 model'])
            table.add_row(['file_4','model_2','obj2 model'])
            table.add_row(['file_5','...','...'])
            st.write(table)
        elif len(file) >= 4:        
            df = pd.read_csv(file[0])
            check_string_NaN(df)
            colored_header(label="Feature and target",description=" ",color_name="violet-70")

            target_num = st.number_input('target number',  min_value=2, max_value=2, value=2)
            
            col_feature, col_target = st.columns(2)
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())
            colored_header(label="target", description=" ", color_name="violet-70")
            target_selected_option = st.multiselect('target', list(targets)[::-1], default=targets.columns.tolist())
            df_var = pd.read_csv(file[1])
            features_name = df_var.columns.tolist()
            range_var = df_var.values
            vars_min = get_column_min(range_var)
            vars_max = get_column_max(range_var)
            array_vars_min = np.array(vars_min).reshape(1,-1)
            array_vars_max = np.array(vars_max).reshape(1,-1)
            vars_bound = np.concatenate([array_vars_min, array_vars_max], axis=0)
            colored_header(label="Feature design constraint", description=" ", color_name="violet-70")
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
                    pareto_front = pareto_front.reset_index(drop=True)
                    st.write(pareto_front)
                    tmp_download_link = download_button(pareto_front, f'Pareto_front.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)

            with st.container():
                button_train = st.button('Opt', use_container_width=True)

            if button_train:               
                plot = customPlot()  
                alg = NSGA2(
                    pop_size=inputs['size pop'],
                    crossover=nsgaSBX(prob=0.9, eta=15),
                    mutation=PM(eta=20),
                    eliminate_duplicates=True
                    )
                if inputs['model'] == 'SMSEMOA':
                    alg = SMSEMOA()

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
                    best_y = pd.DataFrame(best_y, columns = targets.columns.tolist())
                    data = pd.concat([best_x, best_y], axis = 1)
                    st.write(data)               
                    tmp_download_link = download_button(data, f'recommended samples.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                with col2:
                    iter_pareto_front = iter_pareto_front.reset_index(drop=True)
                    st.write(iter_pareto_front)
                    tmp_download_link = download_button(iter_pareto_front, f'iter_pareto_front.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)

    elif sub_option == "Multi-objective Surrogate Optimization (TL)":
        colored_header(label="Multi-objective Surrogate Optimization (TL)",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.pickle` model and `.csv` file",  label_visibility="collapsed", accept_multiple_files=True)
        if len(file) < 6:
            table = PrettyTable(['file name', 'class','description'])
            table.add_row(['file_1','dataset','data file'])
            table.add_row(['file_2','boundary','feature design constraint'])
            table.add_row(['file_3','weights_1','obj1 estimator weight'])
            table.add_row(['file_4','weights_2','obj2 estimator weight'])
            table.add_row(['file_5','model_1','obj1 model1'])
            table.add_row(['file_6','model_2','obj2 model2'])
            st.write(table)
        if len(file) >= 6:
            df = pd.read_csv(file[0])
            check_string_NaN(df)
            colored_header(label="Feature and target",description=" ",color_name="violet-70")

            target_num = st.number_input('target number',  min_value=2, max_value=2, value=2)
            
            col_feature, col_target = st.columns(2)
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())
            colored_header(label="target", description=" ", color_name="violet-70")
            target_selected_option = st.multiselect('target', list(targets)[::-1], default=targets.columns.tolist())
            df_var = pd.read_csv(file[1])
            features_name = df_var.columns.tolist()
            range_var = df_var.values
            vars_min = get_column_min(range_var)
            vars_max = get_column_max(range_var)
            array_vars_min = np.array(vars_min).reshape(1,-1)
            array_vars_max = np.array(vars_max).reshape(1,-1)
            vars_bound = np.concatenate([array_vars_min, array_vars_max], axis=0)
            colored_header(label="Feature design constraint", description=" ", color_name="violet-70")
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
                    pareto_front = pareto_front.reset_index(drop=True)
                    st.write(pareto_front)
                    tmp_download_link = download_button(pareto_front, f'Pareto_front.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)

            with st.container():
                button_train = st.button('Opt', use_container_width=True)

            if button_train:               
                plot = customPlot()  
                alg = NSGA2(
                    pop_size=inputs['size pop'],
                    crossover=nsgaSBX(prob=0.9, eta=15),
                    mutation=PM(eta=20),
                    eliminate_duplicates=True
                    )
                if inputs['model'] == 'SMSEMOA':
                    alg = SMSEMOA()
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
                    best_y = pd.DataFrame(best_y, columns = targets.columns.tolist())
                    data = pd.concat([best_x, best_y], axis = 1)
                    tmp_download_link = download_button(data, f'recommended samples.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                with col2:
                    iter_pareto_front = iter_pareto_front.reset_index(drop=True)
                    st.write(iter_pareto_front)
                    tmp_download_link = download_button(iter_pareto_front, f'iter_pareto_front.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
            
    elif sub_option == "Single-objective Surrogate Optimization (TL)":
        colored_header(label="Single-objective Surrogate Optimization (TL)",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.pickle` model and `.csv` file",  label_visibility="collapsed", accept_multiple_files=True)
        if len(file) < 3:
            table = PrettyTable(['file name', 'class','description'])
            table.add_row(['file_1','boundary','feature design constraint'])
            table.add_row(['file_2','weights','estimator weight'])
            table.add_row(['file_3','model','model'])
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
            colored_header(label="Feature design constraint", description=" ", color_name="violet-70")
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

elif select_option == "Model Inference":
    
    colored_header(label="Model Inference",description=" ",color_name="violet-90")
    file = st.file_uploader("Upload `.csv`file", label_visibility="collapsed", accept_multiple_files=True)
    if len(file) < 2:
        table = PrettyTable(['file name', 'class','description'])
        table.add_row(['file_1','data set (+test data)','data file'])
        table.add_row(['file_2','model','model'])
        st.write(table)
    elif len(file) == 2:
        df = pd.read_csv(file[0])
        model_file = file[1]

        check_string_NaN(df)

        colored_header(label="Data information", description=" ",color_name="violet-70")
        nrow = st.slider("rows", 1, len(df), 5)
        df_nrow = df.head(nrow)
        st.write(df_nrow)

        colored_header(label="Feature and target",description=" ",color_name="violet-70")

        target_num = st.number_input('target number',  min_value=1, max_value=10, value=1)

        col_feature, col_target = st.columns(2)
        # features
        features = df.iloc[:,:-target_num]
        # targets
        targets = df.iloc[:,-target_num:]
        with col_feature:    
            st.write(features.head())
        with col_target:   
            st.write(targets.head())    
        colored_header(label="target", description=" ", color_name="violet-70")

        target_selected_option = st.selectbox('target', list(targets)[::-1])

        targets = targets[target_selected_option]
        preprocess = st.selectbox('data preprocess',[None, 'StandardScaler','MinMaxScaler'])
        if preprocess == 'StandardScaler':
            features = StandardScaler().fit_transform(features)
        elif preprocess == 'MinMaxScaler':
            features = MinMaxScaler().fit_transform(features)

        model = pickle.load(model_file)
        prediction = model.predict(features)
        # st.write(std)
        plot = customPlot()
        plot.pred_vs_actual(targets, prediction)
        r2 = r2_score(targets, prediction)
        st.write('R2: {}'.format(r2))
        result_data = pd.concat([targets, pd.DataFrame(prediction)], axis=1)
        result_data.columns = ['actual','prediction']
        with st.expander('prediction'):
            st.write(result_data)
            tmp_download_link = download_button(result_data, f'prediction.csv', button_text='download')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
        st.write('---')

elif select_option == "Interpretable Machine Learning":
    colored_header(label="Interpretable Machine Learning",description=" ",color_name="violet-90")

    file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
    if file is None:
        table = PrettyTable(['file name', 'class','description'])
        table.add_row(['file_1','dataset','data file'])
        st.write(table)        
    if file is not None:
        df = pd.read_csv(file)
        check_string_NaN(df)
        colored_header(label="Data information", description=" ",color_name="violet-70")
        nrow = st.slider("rows", 1, len(df), 5)
        df_nrow = df.head(nrow)
        st.write(df_nrow)

        colored_header(label="Feature and target",description=" ",color_name="violet-70")

        target_num = st.number_input('target number',  min_value=1, max_value=10, value=1)
        
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
        random_state = st.checkbox('random state 42',True)
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



