import importlib.util

import streamlit as st

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition.alloy import WenAlloys
from matminer.featurizers.composition import ElementProperty

from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt

import collections
import os

# utilities
import base64
import pickle
import uuid
import re

import scienceplots

# ============ import model from file =======================

def import_from_file(module_name: str, filepath: str):
    """
    Imports a module from file.

    Args:
        module_name (str): Assigned to the module's __name__ parameter (does not 
            influence how the module is named outside of this function)
        filepath (str): Path to the .py file

    Returns:
        The module
    """
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============== model selected =======================

def model_platform(model_path):
    template_dict = collections.defaultdict(dict)
    template_dirs = [f for f in os.scandir(model_path) if f.is_dir() and f.name != "example"]
    # Todo: Find a good way to sort templates, e.g. vy prepending a number to their name
    # (e.g. 1_Image classification_PyTorch).

    template_dirs = sorted(template_dirs, key = lambda e:e.name)

    for template_dir in template_dirs:
        try: 
            # st.write(template_dir)
            # Template with task + framework.
            task, framework = template_dir.name.split("_")
            template_dict[task][framework] = template_dir.path
        except ValueError:
            # Templates with task only.
            template_dict[template_dir.name] = template_dir.path

    task = st.selectbox("model", list(template_dict.keys()))
    if isinstance(template_dict[task], dict):
        framework = st.selectbox('platform', list(template_dict[task].keys()))
        template_dir = template_dict[task][framework]
    else:
        template_dir = template_dict[task]

    template_alg = import_from_file(
    "template_alg", os.path.join(template_dir, "alg.py")
    )
    return template_alg

# =============== some tools =======================
def create_data_with_group_and_counts(feature_type):
    comp_ids = []
    comp_counts = []
    for name,group in feature_type:
        comp_ids.append(name)
        comp_counts.append(len(group))
    comp_instances= list(zip(comp_ids,comp_counts))
    comp_data = pd.DataFrame(comp_instances,columns=['ID','Count']).sort_values(by='Count')
    return comp_data
# ============== check string in DataFrame ============
def check_string_NaN(df):
    # check NaN
    null_columns = df.columns[df.isnull().any()]
    if len(null_columns) > 0:
        st.error(f"Error: NaN in column {list(null_columns)} !")
        st.stop()
    # check string
    string_columns = df.select_dtypes(include=[object]).columns
    string_contains_columns = []
    for column in string_columns:
        if df[column].astype(str).str.contains('').any():
            string_contains_columns.append(column)
    if len(string_contains_columns) > 0:
        st.error(f"Error: string in column {string_contains_columns} !")
        st.stop()

def check_string(df):
    null_columns = df.columns[df.isnull().any()]
    if len(null_columns) > 0:
        st.error(f"Error: NaN in column {list(null_columns)} !")
        st.stop()
    string_columns = df.select_dtypes(include=[object]).columns
    string_contains_columns = []
    for column in string_columns:
        if df[column].astype(str).str.contains('').any():
            string_contains_columns.append(column)
    if len(string_contains_columns) == 0:
        st.error(f"Error: need string in label column!")
        st.stop()
# =============== dwonload button =======================  
def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None
    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = pickle.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    prim_color = '#ffffff' 
    bg_color1 = '#66CDAA'
    bg_color2 = '#66CDAA'
    sbg_color = '#111111'
    txt_color = '#111111' 
    font = 'Microsoft YaHei'  


    custom_css = f"""
        <style>
            #{button_id} {{
                background-color: {bg_color1};
                color: {txt_color};
                padding: 0.25rem 0.75rem;
                position: relative;
                line-height: 1.6;
                border-radius: 0.25rem;
                border-width: 1px;
                border-style: solid;
                border-color: #ffffff;
                border-image: initial;
                filter: brightness(105%);
                justify-content: center;
                margin: 0px;
                width: auto;
                appearance: button;
                display: inline-flex;
                family-font: {font};
                font-weight: 400;
                letter-spacing: normal;
                word-spacing: normal;
                text-align: center;
                text-rendering: auto;
                text-transform: none;
                text-indent: 0px;
                text-shadow: none;
                text-decoration: none;
            }}
            #{button_id}:hover {{

                border-color: {prim_color};
                color: #ffff00;
                background-color: {bg_color2};
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: #ff0066;
                color: {sbg_color};
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" class= "" id="{button_id}" ' \
                           f'href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link

# =============== class =======================

class customPlot:
    def __init__(self) -> None:
        self.fig_title = None
        self.fig_label = None
        self.fig_legend = None
        self.fontsize_option = ('MEDIUM','SMALL','BIGGER','SSMALL','SSSMALL')
        self.fontsize_dict = {'SSSMALL':3,'SSMALL':5,'SMALL':8, 'MEDIUM':12, 'BIGGER':17}
        self.color_option = ('MidnightBlue',
                            'Pink',
                            'Crimson',
                            'LavenderBlush',
                            'PaleVioletRed',
                            'HotPink',
                            'DeepPink',
                            'MediumVioletRed',
                            'Orchid',
                            'Thistle',
                            'plum',
                            'Violet',
                            'Magenta',
                            'Fuchsia',
                            'DarkMagenta',
                            'Purple',
                            'MediumOrchid',
                            'DarkVoilet',
                            'DarkOrchid',
                            'Indigo',
                            'BlueViolet',
                            'MediumPurple',
                            'MediumSlateBlue',
                            'SlateBlue',
                            'DarkSlateBlue',
                            'Lavender',
                            'GhostWhite',
                            'Blue',
                            'MediumBlue',
                            'LightPink',
                            'DarkBlue',
                            'Navy',
                            'RoyalBlue',
                            'CornflowerBlue',
                            'LightSteelBlue',
                            'LightSlateGray',
                            'SlateGray',
                            'DoderBlue',
                            'AliceBlue',
                            'SteelBlue',
                            'LightSkyBlue',
                            'SkyBlue',
                            'DeepSkyBlue',
                            'LightBLue',
                            'PowDerBlue',
                            'CadetBlue',
                            'Azure',
                            'LightCyan',
                            'PaleTurquoise',
                            'Cyan',
                            'Aqua',
                            'DarkTurquoise',
                            'DarkSlateGray',
                            'DarkCyan',
                            'Teal',
                            'MediumTurquoise',
                            'LightSeaGreen',
                            'Turquoise',
                            'Auqamarin',
                            'MediumAquamarine',
                            'MediumSpringGreen',
                            'MintCream',
                            'SpringGreen',
                            'SeaGreen',
                            'Honeydew',
                            'LightGreen',
                            'PaleGreen',
                            'DarkSeaGreen',
                            'LimeGreen',
                            'Lime',
                            'ForestGreen',
                            'Green',
                            'DarkGreen',
                            'Chartreuse',
                            'LawnGreen',
                            'GreenYellow',
                            'OliveDrab',
                            'Beige',
                            'LightGoldenrodYellow',
                            'Ivory',
                            'LightYellow',
                            'Yellow',
                            'Olive',
                            'DarkKhaki',
                            'LemonChiffon',
                            'PaleGodenrod',
                            'Khaki',
                            'Gold',
                            'Cornislk',
                            'GoldEnrod',
                            'FloralWhite',
                            'OldLace',
                            'Wheat',
                            'Moccasin',
                            'Orange',
                            'PapayaWhip',
                            'BlanchedAlmond',
                            'NavajoWhite',
                            'AntiqueWhite',
                            'Tan',
                            'BrulyWood',
                            'Bisque',
                            'DarkOrange',
                            'Linen',
                            'Peru',
                            'PeachPuff',
                            'SandyBrown',
                            'Chocolate',
                            'SaddleBrown',
                            'SeaShell',
                            'Sienna',
                            'LightSalmon',
                            'Coral',
                            'OrangeRed',
                            'DarkSalmon',
                            'Tomato',
                            'MistyRose',
                            'Salmon',
                            'Snow',
                            'LightCoral',
                            'RosyBrown',
                            'IndianRed',
                            'Red',
                            'Brown',
                            'FireBrick',
                            'DarkRed',
                            'Maroon',
                            'White',
                            'WhiteSmoke',
                            'Gainsboro',
                            'LightGrey',
                            'Silver',
                            'DarkGray',
                            'Gray',
                            'DimGray',
                            'Black')
        self.color_dict = {'MidnightBlue':'#191970',
                        'Pink':'#FFC0CB',
                        'Crimson':'#DC143C',
                        'LavenderBlush':'#FFF0F5',
                        'PaleVioletRed':'#DB7093',
                        'HotPink':'#FF69B4',
                        'DeepPink':'#FF1493',
                        'MediumVioletRed':'#C71585',
                        'Orchid':'#DA70D6',
                        'Thistle':'#D8BFD8',
                        'plum':'#DDA0DD',
                        'Violet':'#EE82EE',
                        'Magenta':'#FF00FF',
                        'Fuchsia':'#FF00FF',
                        'DarkMagenta':'#8B008B',
                        'Purple':'#800080',
                        'MediumOrchid':'#BA55D3',
                        'DarkVoilet':'#9400D3',
                        'DarkOrchid':'#9932CC',
                        'Indigo':'#4B0082',
                        'BlueViolet':'#8A2BE2',
                        'MediumPurple':'#9370DB',
                        'MediumSlateBlue':'#7B68EE',
                        'SlateBlue':'#6A5ACD',
                        'DarkSlateBlue':'#483D8B',
                        'Lavender':'#E6E6FA',
                        'GhostWhite':'#F8F8FF',
                        'Blue':'#0000FF',
                        'MediumBlue':'#0000CD',
                        'LightPink':'#FFB6C1',
                        'DarkBlue':'#00008B',
                        'Navy':'#000080',
                        'RoyalBlue':'#4169E1',
                        'CornflowerBlue':'#6495ED',
                        'LightSteelBlue':'#B0C4DE',
                        'LightSlateGray':'#778899',
                        'SlateGray':'#708090',
                        'DoderBlue':'#1E90FF',
                        'AliceBlue':'#F0F8FF',
                        'SteelBlue':'#4682B4',
                        'LightSkyBlue':'#87CEFA',
                        'SkyBlue':'#87CEEB',
                        'DeepSkyBlue':'#00BFFF',
                        'LightBLue':'#ADD8E6',
                        'PowDerBlue':'#B0E0E6',
                        'CadetBlue':'#5F9EA0',
                        'Azure':'#F0FFFF',
                        'LightCyan':'#E1FFFF',
                        'PaleTurquoise':'#AFEEEE',
                        'Cyan':'#00FFFF',
                        'Aqua':'#D4F2E7',
                        'DarkTurquoise':'#00CED1',
                        'DarkSlateGray':'#2F4F4F',
                        'DarkCyan':'#008B8B',
                        'Teal':'#008080',
                        'MediumTurquoise':'#48D1CC',
                        'LightSeaGreen':'#20B2AA',
                        'Turquoise':'#40E0D0',
                        'Auqamarin':'#7FFFAA',
                        'MediumAquamarine':'#00FA9A',
                        'MediumSpringGreen':'#00FF7F',
                        'MintCream':'#F5FFFA',
                        'SpringGreen':'#3CB371',
                        'SeaGreen':'#2E8B57',
                        'Honeydew':'#F0FFF0',
                        'LightGreen':'#90EE90',
                        'PaleGreen':'#98FB98',
                        'DarkSeaGreen':'#8FBC8F',
                        'LimeGreen':'#32CD32',
                        'Lime':'#00FF00',
                        'ForestGreen':'#228B22',
                        'Green':'#008000',
                        'DarkGreen':'#006400',
                        'Chartreuse':'#7FFF00',
                        'LawnGreen':'#7CFC00',
                        'GreenYellow':'#ADFF2F',
                        'OliveDrab':'#556B2F',
                        'Beige':'#F5F5DC',
                        'LightGoldenrodYellow':'#FAFAD2',
                        'Ivory':'#FFFFF0',
                        'LightYellow':'#FFFFE0',
                        'Yellow':'#FFFF00',
                        'Olive':'#808000',
                        'DarkKhaki':'#BDB76B',
                        'LemonChiffon':'#FFFACD',
                        'PaleGodenrod':'#EEE8AA',
                        'Khaki':'#F0E68C',
                        'Gold':'#FFD700',
                        'Cornislk':'#FFF8DC',
                        'GoldEnrod':'#DAA520',
                        'FloralWhite':'#FFFAF0',
                        'OldLace':'#FDF5E6',
                        'Wheat':'#F5DEB3',
                        'Moccasin':'#FFE4B5',
                        'Orange':'#FFA500',
                        'PapayaWhip':'#FFEFD5',
                        'BlanchedAlmond':'#FFEBCD',
                        'NavajoWhite':'#FFDEAD',
                        'AntiqueWhite':'#FAEBD7',
                        'Tan':'#D2B48C',
                        'BrulyWood':'#DEB887',
                        'Bisque':'#FFE4C4',
                        'DarkOrange':'#FF8C00',
                        'Linen':'#FAF0E6',
                        'Peru':'#CD853F',
                        'PeachPuff':'#FFDAB9',
                        'SandyBrown':'#F4A460',
                        'Chocolate':'#D2691E',
                        'SaddleBrown':'#8B4513',
                        'SeaShell':'#FFF5EE',
                        'Sienna':'#A0522D',
                        'LightSalmon':'#FFA07A',
                        'Coral':'#FF7F50',
                        'OrangeRed':'#FF4500',
                        'DarkSalmon':'#E9967A',
                        'Tomato':'#FF6347',
                        'MistyRose':'#FFE4E1',
                        'Salmon':'#FA8072',
                        'Snow':'#FFFAFA',
                        'LightCoral':'#F08080',
                        'RosyBrown':'#BC8F8F',
                        'IndianRed':'#CD5C5C',
                        'Red':'#FF0000',
                        'Brown':'#A52A2A',
                        'FireBrick':'#B22222',
                        'DarkRed':'#8B0000',
                        'Maroon':'#800000',
                        'White':'#FFFFFF',
                        'WhiteSmoke':'#F5F5F5',
                        'Gainsboro':'#DCDCDC',
                        'LightGrey':'#D3D3D3',
                        'Silver':'#C0C0C0',
                        'DarkGray':'#A9A9A9',
                        'Gray':'#808080',
                        'DimGray':'#696969',
                        'Black':'#000000'}
    def set_title_fontsize(self, key):
        options = self.fontsize_option
        options_selected = st.selectbox('title fontsize', options, key=key)
        return options_selected
    def set_label_fontsize(self, key):
        options = self.fontsize_option
        options_selected = st.selectbox('label fontsize', options, key=key)
        return options_selected
    def set_annot_fontsize(self, key):
        options = self.fontsize_option
        options_selected = st.selectbox('annot fontsize', options, key=key)
        return options_selected       
    def set_legend_fontsize(self, key):
        options = self.fontsize_option
        options_selected = st.selectbox('legend fontsize', options, key=key)
        return options_selected
    
    def set_tick_fontsize(self, key):
        options = self.fontsize_option
        options_selected = st.selectbox('tick fontsize', options, key=key)
        return options_selected
    
    def set_color(self, name, num, key):
        options = self.color_option
        options_selected = st.selectbox(name, options,num,key=key)
        return options_selected
    
    def map_fontsize_options(self, options_selected):
        options = np.array([self.fontsize_dict[x] for x in options_selected[:4]])
        plt.rc('axes', titlesize=options[0])     
        plt.rc('axes', labelsize=options[1])    
        plt.rc('xtick', labelsize=options[2])  
        plt.rc('ytick', labelsize=options[2])    
        plt.rc('legend', fontsize=options[3], frameon=False) 
    
    def map_color_options(self, options_selected):
        color = [self.color_dict[x] for x in options_selected[4:]]
        return color
    
    def target_hist_kde(self, options_selected, target_name, target_value):
        assert len(options_selected) >= 4, "options insufficient !"
        self.map_fontsize_options(options_selected)
        color = self.map_color_options(options_selected)
        fig, ax = plt.subplots()  
        ax = sns.distplot(target_value.dropna(), bins=10, kde_kws={"color": color[0], "label": "KDE","alpha":1},
                            hist_kws={"color": color[1],"alpha":1})
        plt.title(target_name + " Statistics ")
        plt.legend()

        st.pyplot(fig)

    def feature_hist_kde(self, options_selected, feature_name, feature_value):
        assert len(options_selected) >= 4, "options insufficient !"
        self.map_fontsize_options(options_selected)
        color = self.map_color_options(options_selected)
        fig, ax = plt.subplots()
        ax = sns.distplot(feature_value.dropna(), bins=10, kde_kws={"color": color[0], "label": "KDE","alpha":1},
                            hist_kws={"color": color[1],"alpha":1})
        plt.title(feature_name + " Statistics ")
        plt.legend()

        st.pyplot(fig)

    def featureSets_statistics_hist(self, options_selected, IDs, Counts):
        assert len(options_selected) >= 4, "options insufficient !"
        self.map_fontsize_options(options_selected)
        color = self.map_color_options(options_selected)
        fig, ax = plt.subplots()
        ax = plt.bar(IDs, Counts,color=color[0],label='Count')
        plt.title(" Recipes Statistics ")
        plt.xticks(IDs, rotation='vertical')
        plt.xlabel("Recipes")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.legend()

        st.pyplot(fig)

    def feature_distribution(self,options_selected,feature_name, feature_value):
        assert len(options_selected) >= 4, "options insufficient !"
        self.map_fontsize_options(options_selected)
        color = self.map_color_options(options_selected)
        fig, ax = plt.subplots()
        plt.hist(x=feature_value, bins='auto', color=color,alpha=0.7, rwidth=0.85)
        number = (feature_value != 0).values.sum()
        occupy_ratio = int(number/len(feature_value) * 100)
        number_string = str(number) + " data points"+'\n'+ str(occupy_ratio)+ "% of data"
        plt.text(0.1, 0.8, number_string, transform=ax.transAxes, fontdict={'color': 'black'})
        plt.title("Distribution of "+ feature_name)
        plt.ylabel("Frequency")
        plt.xlabel("Atomic Percentage (at.%)")
        plt.legend()

        st.pyplot(fig)

    def features_and_targets(self,options_selected, data, features, targets):
        assert len(options_selected) >= 4, "options insufficient !"
        self.map_fontsize_options(options_selected)
        color = self.map_color_options(options_selected)
        fig = sns.PairGrid(data, x_vars=features, y_vars=targets).map(sns.regplot,ci=None,scatter_kws={"color": color[0]}, line_kws={"color": color[1]})
        plt.legend()

        st.pyplot(fig)

    def targets_and_targets(self,options_selected, data, targets):
        assert len(options_selected) >= 4, "options insufficient !"
        self.map_fontsize_options(options_selected)
        color = self.map_color_options(options_selected)
        fig = sns.PairGrid(data, x_vars=targets, y_vars=targets, height=2, aspect=1.5).map(sns.regplot,ci=None,scatter_kws={"color": color[0]}, line_kws={"color": color[1]})
        for i, j in zip(*np.triu_indices_from(fig.axes, 1)):
            fig.axes[i, j].set_visible(False)
        for i in range(len(targets)): 
            fig.axes[i, i].set_visible(False)
        plt.legend()

        st.pyplot(fig)

    def corr_feature_target(self, options_selected, corr):
        assert len(options_selected) >= 4, "options insufficient !"
        self.map_fontsize_options(options_selected)
        color = self.map_color_options(options_selected)
        fig, ax = plt.subplots()
        ax = plt.barh(corr.index, corr.values, color=color)
        plt.xlabel('Correlation')
        plt.title('Feature Correlations')
        plt.legend()

        st.pyplot(fig)
    def corr_feature_target_mir(self, options_selected, corr_mir):
        assert len(options_selected) >= 4, "options insufficient !"
        self.map_fontsize_options(options_selected)
        color = self.map_color_options(options_selected)
        fig, ax = plt.subplots()
        ax = plt.barh(corr_mir.index, corr_mir['mutual info'], color=color)
        plt.xlabel('Correlation')
        plt.title('Feature Correlations')
        plt.legend()
        st.pyplot(fig)

    def corr_cofficient(self, options_selected, is_mask, corr_matrix):
        options = np.array([self.fontsize_dict[x] for x in options_selected[:4]])
        plt.rc('xtick', labelsize=options[0])  
        plt.rc('ytick', labelsize=options[0])    
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        annot_kws = {"size": options[1]}
        fig, ax = plt.subplots()
        if is_mask == "Yes":
            mask = np.zeros_like(corr_matrix, dtype=bool)
            mask[np.triu_indices_from(mask)] = True
            ax = sns.heatmap(corr_matrix, mask=mask, cmap=cmap, linewidths=0.5,square = True, annot=True, annot_kws=annot_kws)
            plt.legend()

            st.pyplot(fig)
        else:
            ax = sns.heatmap(corr_matrix,cmap = cmap, linewidths=0.5,square = True, annot=True, annot_kws=annot_kws)
            plt.legend()

            st.pyplot(fig)

    def feature_missing(self,options_selected, record_missing, missing_stats):
        # Histogram of missing fraction in each features
        if record_missing is None:
            raise NotImplementedError("Missing values have not been calculated.")
        assert len(options_selected) >= 4, "options insufficient !"
        self.map_fontsize_options(options_selected)
        color = self.map_color_options(options_selected)
        fig, ax = plt.subplots()
        ax = plt.hist(missing_stats['missing_fraction'], color=color[0], align='left')
        plt.title(" Fraction of Missing Values Histogram ")
        plt.xlabel("Missing Fraction")
        plt.ylabel("Count of Features")
        plt.tight_layout()
        plt.legend()

        st.pyplot(fig)    

    def feature_nunique(self, options_selected, record_single_unique, unique_stats):
        if record_single_unique is None:
            raise NotImplementedError("Unique values have not been calculated.")
        assert len(options_selected) >= 4, "options insufficient !"
        self.map_fontsize_options(options_selected)
        color = self.map_color_options(options_selected)
        fig, ax = plt.subplots()
        ax = plt.hist(unique_stats, color=color[0], align = "left")
        plt.title('Number of Unique Values Histogram')
        plt.xlabel('Unique Values')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.legend()

        st.pyplot(fig)    

    def feature_importance(self,record_zero_importance, feature_importances, plot_n = 15):

        if record_zero_importance is None:
            raise NotImplementedError("Unique values have not been calculated.")

        if plot_n > feature_importances.shape[0]:
            plot_n = feature_importances.shape[0] 

        with plt.style.context(['nature','no-latex']):
            fig, ax = plt.subplots()
            ax.barh(list(reversed(list(feature_importances.index[:plot_n]))), 
                feature_importances['normalized_importance'][:plot_n], 
                align = 'center', edgecolor = 'k')

            ax.set_yticks(list(reversed(list(feature_importances.index[:plot_n]))))
            ax.set_yticklabels(feature_importances['feature'][:plot_n])
            
            plt.xlabel('Normalized Importance'); 
            plt.title('Feature Importances')
            plt.legend()

            st.pyplot(fig)   

    def pred_vs_actual(self,actual,pred):
        with plt.style.context(['nature','no-latex']):
            fig, ax = plt.subplots()  
            ax.scatter(actual, pred, marker='o', s=18, color='#000080',zorder=1, facecolors='none')
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                    ]
            ax.tick_params(direction='in', length=5)
    
            ax.plot(lims, lims, 'k-', zorder=2, linewidth=2, linestyle='solid', color='#FF0000')
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            plt.xlabel("Actual")
            plt.ylabel("Prediction")
            st.pyplot(fig)

    def confusion_matrix(self,confusion_matrix):
        
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        fig, ax = plt.subplots()  
        ax = sns.heatmap(confusion_matrix, cmap=cmap, linewidths=0.5,square =True, annot=True, xticklabels=df['Category2'].unique())
        plt.ylabel("Actual")
        plt.xlabel("Predict")

        st.pyplot(fig)
    def evolutionary_history(self, fitness_history, algName):
        fig, ax = plt.subplots()
        ax = plt.plot(np.array(fitness_history), label=algName)
        plt.title(" Evolutionary history ")
        plt.xlabel("Genetation")
        plt.ylabel("Fitness")
        plt.tight_layout()
        plt.legend()
        st.pyplot(fig)       


class FeatureSelector:
    def __init__(self, features, targets) -> None:
        
        # origin features and targets
        self.features = features
        self.targets = targets
        
        # model
        self.model = None

        if targets is None:
            st.write('No targets provided !')
        # origin features name 
        self.base_features = list(features.columns)

        # one-hot features name
        self.one_hot_features = None

        self.record_missing = None
        self.record_single_unique = None
        self.record_collinear = None
        self.record_zero_importance = None
        self.record_low_importance = None

        # Dataframe recording the different features
        self.features_dropped_single = None
        self.features_plus_oneHot = None
        self.dummy_features = None
        self.features_dropped_f_t = None
        self.features_dropped_collinear = None
        self.features_dropped_low_importance = None
        self.features_dropped_zero_importance = None

        self.missing_stats = None
        self.unique_stats = None
        self.corr_matrix = None
        self.feature_importances = None
        
        # Dictionary to hold removal operations
        self.ops = {}
        
        self.one_hot_correlated = False  
        
    
    def identify_missing(self, missing_threshold):

        self.missing_threshold = missing_threshold

        missing_series = self.features.isnull().sum() / self.features.shape[0]
        self.missing_stats = pd.DataFrame(missing_series).rename(columns={'index': 'feature', 0: 'missing_fraction'})

        self.missing_stats = self.missing_stats.sort_values('missing_fraction', ascending = False)

        record_missing = pd.DataFrame(missing_series[missing_series > missing_threshold]).reset_index().rename(
            columns={'index':'feature', 0: 'missing_fraction'})
        to_drop = list(record_missing['feature'])
        self.record_missing = record_missing
        self.ops['missing'] = to_drop
    
    def identify_nunique(self, counts=1):

        self.counts = counts
        unique_counts = self.features.nunique()
        self.unique_stats = pd.DataFrame(unique_counts).rename(columns = {'index': 'feature', 0: 'nunique'})
        self.unique_stats = self.unique_stats.sort_values('nunique', ascending = True)

        record_single_unique = pd.DataFrame(unique_counts[unique_counts <= self.counts]).reset_index().rename(
            columns = {'index':'feature', 0: 'nunique'})
        
        to_drop = list(record_single_unique['feature'])

        self.record_single_unique = record_single_unique
        self.ops['single_unique'] = to_drop
        
    def one_hot_feature_encoder(self,one_hot=False):

        if one_hot:
            # One hot encoding
            self.dummy_features = pd.get_dummies(self.features)
            self.one_hot_features = [column for column in self.dummy_features.columns if column not in self.base_features]
            # Add one hot encoded data to original data
            self.features_plus_oneHot = pd.concat([self.dummy_features[self.one_hot_features], self.features], axis=1)
    

    def identify_collinear(self, corr_matrix, correlation_threshold):

        self.correlation_threshold = correlation_threshold

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

        record_collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature','corr_value'])

        for column in to_drop:

            corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

            corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]

            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                              'corr_feature': corr_features,
                                              'corr_value':corr_values})
            record_collinear = record_collinear._append(temp_df, ignore_index=True)
        self.record_collinear = record_collinear

    def judge_drop_f_t_after_f_f(self, target_selected, corr_matrix):

        assert len(target_selected) == 1, 'please choose only one feature !' 

        for row in range(self.record_collinear.shape[0]):
            need_judge_feature_idx = list(self.record_collinear[['drop_feature','corr_feature']].iloc[row,:])
            need_judge_feature_value = corr_matrix.loc[need_judge_feature_idx, target_selected]   
            feature_not_drop_index = list(need_judge_feature_value.abs().idxmax())
            feature_need_drop_index = list(need_judge_feature_value.abs().idxmin())
            self.record_collinear.loc[row,'corr_feature'] = pd.DataFrame(feature_not_drop_index).loc[0,0]
            self.record_collinear.loc[row,'drop_feature'] = pd.DataFrame(feature_need_drop_index).loc[0,0]
        self.ops['collinear'] = self.record_collinear['drop_feature']

    #   corr_threshold  
    def judge_drop_f_t(self, target_selected, corr_matrix, corr_threshold):
        
        assert len(target_selected) == 1, 'please choose only one feature !' 
        f_t =  corr_matrix.loc[target_selected]
        less_than_thresh = f_t.abs().lt(corr_threshold)
        cols_with_less_than_thresh = less_than_thresh.any()
        self.ops['f_t_low_corr'] = cols_with_less_than_thresh[cols_with_less_than_thresh == True].index.tolist()    

    def identify_zero_low_importance(self, cumulative_importance=0.95):
   
        feature_importances = self.feature_importances.sort_values('importance', ascending=False).reset_index(drop = True)

        feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
        feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])

        record_zero_importance = feature_importances[feature_importances['importance'] == 0.0]
        to_drop = list(record_zero_importance['feature'])
        self.feature_importances = feature_importances
        self.record_zero_importance = record_zero_importance
        self.ops['zero_importance'] = to_drop
        st.write('\n%d features with zero importance.\n' % len(self.ops['zero_importance']))

        self.cumulative_importance = cumulative_importance

        record_low_importance = self.feature_importances[self.feature_importances['cumulative_importance'] > cumulative_importance]
        to_drop = list(record_low_importance['feature'])
        self.record_low_importance = record_low_importance
        self.ops['low_importance'] = to_drop
        st.write('%d features required for cumulative importance of %0.2f .' % (len(self.feature_importances) -
                                                                        len(self.record_low_importance), self.cumulative_importance))
    def feature_importance_select_show(self):
        st.write('---')
        st.write(self.feature_importances)
        tmp_download_link = download_button(self.feature_importances, f'importance rank.csv', button_text='download')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
        plot = customPlot()
        plot.feature_importance(self.record_zero_importance, self.feature_importances, plot_n = 15)
        self.features_dropped_zero_importance = self.features.drop(columns=self.ops['zero_importance'])
        self.features_dropped_low_importance = self.features.drop(columns=self.ops['low_importance'])
        with st.expander('dropped data'):
            col1, col2 = st.columns([1,1])
            with col1:
                st.write('dropped zero importance')
                st.write(self.features_dropped_zero_importance)
                tmp_download_link = download_button(self.features_dropped_zero_importance, f'featureImportance_drop_0.csv', button_text='download')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
            with col2:
                st.write('dropped low importance')
                st.write(self.features_dropped_low_importance)
                tmp_download_link = download_button(self.features_dropped_low_importance, f'featureImportance_drop_low.csv', button_text='download')
                st.markdown(tmp_download_link, unsafe_allow_html=True)

    def LinearRegressor(self):

        features = pd.get_dummies(self.features)

        feature_names = list(features.columns)

        features = np.array(self.features)
 
        feature_importance_values = np.zeros(len(feature_names))

        self.model.fit(self.features, self.targets)

        feature_importance_values = abs(self.model.coef_)
        self.feature_importances = pd.DataFrame({'feature': feature_names,'importance':feature_importance_values})
    
    def RidgeRegressor(self):

        features = pd.get_dummies(self.features)
    
        feature_names = list(features.columns)

        features = np.array(self.features)

        feature_importance_values = np.zeros(len(feature_names))

        self.model.fit(self.features, self.targets)

        feature_importance_values = abs(self.model.coef_)
        self.feature_importances = pd.DataFrame({'feature': feature_names,'importance':feature_importance_values})

    def LassoRegressor(self):

        features = pd.get_dummies(self.features)

        feature_names = list(features.columns)

        features = np.array(self.features)

        feature_importance_values = np.zeros(len(feature_names))

        self.model.fit(self.features, self.targets)

        feature_importance_values = abs(self.model.coef_)
        self.feature_importances = pd.DataFrame({'feature': feature_names,'importance':feature_importance_values})
    
    def XGBR(self):

        features = pd.get_dummies(self.features)

        feature_names = list(features.columns)

        features = np.array(self.features)
        targets = np.array(self.targets).reshape((-1,))

        feature_importance_values = np.zeros(len(feature_names))

        progress_text = "Operation in progress. Please wait."

        self.model.fit(self.features, self.targets)

        # st.info('train process is over')
        feature_importance_values = abs(self.model.ranking_)
        self.feature_importances = pd.DataFrame({'feature': feature_names,'importance':feature_importance_values})     
          
    def RandomForestClassifier(self):

        features = pd.get_dummies(self.features)

        feature_names = list(features.columns)

        features = np.array(self.features)

        feature_importance_values = np.zeros(len(feature_names))

        progress_text = "Operation in progress. Please wait."

        self.model.fit(self.features, self.targets.astype('int'))

        # st.info('train process is over')
        feature_importance_values = abs(self.model.feature_importances_)
        self.feature_importances = pd.DataFrame({'feature': feature_names,'importance':feature_importance_values})  
    
    def RandomForestRegressor(self):
        features = pd.get_dummies(self.features)

        feature_names = list(features.columns)

        features = np.array(self.features)

        feature_importance_values = np.zeros(len(feature_names))

        self.model.fit(self.features, self.targets.astype('int'))

        feature_importance_values = abs(self.model.feature_importances_)

        self.feature_importances = pd.DataFrame({'feature': feature_names,'importance':feature_importance_values})  

class CLASSIFIER:
    def __init__(self, features, targets) -> None:

        self.features = features
        self.targets = targets   

        self.model = None

        self.score = None

        self.Ypred = None
        
    def DecisionTreeClassifier(self):
        columns = self.Ytest.columns
        self.Ytest = self.Ytest.reset_index(drop=True)
    
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)

        self.Ypred = self.model.predict(self.Xtest)
        self.Ypred = pd.DataFrame(self.Ypred, columns=columns)
        # st.info('train process is over')
        self.score = accuracy_score(self.Ypred ,self.Ytest)
        st.write('accuracy score: {}'.format(self.score))
    
    def RandomForestClassifier(self):
        columns = self.Ytest.columns
        self.Ytest = self.Ytest.reset_index(drop=True)
    
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)

        self.Ypred = self.model.predict(self.Xtest)
        self.Ypred = pd.DataFrame(self.Ypred, columns=columns)
        # st.info('train process is over')
        self.score = accuracy_score(self.Ypred ,self.Ytest)
        st.write('accuracy score: {}'.format(self.score))
        


    def LogisticRegreesion(self):
        columns = self.Ytest.columns
        self.Ytest = self.Ytest.reset_index(drop=True)
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')
     
        self.model.fit(self.Xtrain, self.Ytrain)
        self.Ypred = self.model.predict(self.Xtest)
        self.Ypred = pd.DataFrame(self.Ypred, columns=columns)

        # st.info('train process is over')
        self.score = accuracy_score(self.Ypred ,self.Ytest)
        st.write('accuracy score: {}'.format(self.score))

    def SupportVector(self):
        columns = self.Ytest.columns
        self.Ytest = self.Ytest.reset_index(drop=True)
    
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)

        self.Ypred = self.model.predict(self.Xtest)

        self.Ypred = pd.DataFrame(self.Ypred, columns=columns)
        # st.info('train process is over')
        self.score = accuracy_score(self.Ypred ,self.Ytest)
        st.write('accuracy score: {}'.format(self.score))

    def BaggingClassifier(self):
        columns = self.Ytest.columns
        self.Ytest = self.Ytest.reset_index(drop=True)
    
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)

        self.Ypred = self.model.predict(self.Xtest)
        self.Ypred = pd.DataFrame(self.Ypred, columns=columns)
        # st.info('train process is over')
        self.score = accuracy_score(self.Ypred ,self.Ytest)
        st.write('accuracy score: {}'.format(self.score))    
    
    def AdaBoostClassifier(self):
        columns = self.Ytest.columns
        self.Ytest = self.Ytest.reset_index(drop=True)
    
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)

        self.Ypred = self.model.predict(self.Xtest)
        self.Ypred = pd.DataFrame(self.Ypred, columns=columns)
        # st.info('train process is over')
        self.score = accuracy_score(self.Ypred ,self.Ytest)
        st.write('accuracy score: {}'.format(self.score))  

    def GradientBoostingClassifier(self):
        columns = self.Ytest.columns
        self.Ytest = self.Ytest.reset_index(drop=True)
    
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)

        self.Ypred = self.model.predict(self.Xtest)
        self.Ypred = pd.DataFrame(self.Ypred, columns=columns)
        # st.info('train process is over')
        self.score = accuracy_score(self.Ypred ,self.Ytest)
        st.write('accuracy score: {}'.format(self.score))   
    
    def XGBClassifier(self):
        columns = self.Ytest.columns
        self.Ytest = self.Ytest.reset_index(drop=True)
    
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')


        self.model.fit(self.Xtrain, self.Ytrain)

        self.Ypred = self.model.predict(self.Xtest)
        self.Ypred = pd.DataFrame(self.Ypred, columns=columns)
        # st.info('train process is over')
        self.score = accuracy_score(self.Ypred ,self.Ytest)
        st.write('accuracy score: {}'.format(self.score))  

    def LGBMClassifier(self):
        columns = self.Ytest.columns
        self.Ytest = self.Ytest.reset_index(drop=True)
    
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)

        self.Ypred = self.model.predict(self.Xtest)
        self.Ypred = pd.DataFrame(self.Ypred, columns=columns)
        # st.info('train process is over')
        self.score = accuracy_score(self.Ypred ,self.Ytest)
        st.write('accuracy score: {}'.format(self.score))  

    def CatBoostClassifier(self):
        columns = self.Ytest.columns
        self.Ytest = self.Ytest.reset_index(drop=True)
    
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)

        self.Ypred = self.model.predict(self.Xtest)
        self.Ypred = pd.DataFrame(self.Ypred, columns=columns)
        # st.info('train process is over')
        self.score = accuracy_score(self.Ypred ,self.Ytest)
        st.write('accuracy score: {}'.format(self.score))  

class REGRESSOR:
    def __init__(self, features, targets) -> None:
        # origin features and targets
        self.features = features
        self.targets = targets   

        self.model = None
        self.Ypred = None
        self.score = None
        self.PvsT = None    

    def DecisionTreeRegressor(self):
        self.Ytest = self.Ytest.reset_index(drop=True)
        self.Xtest = self.Xtest.reset_index(drop=True)
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)
        # # st.info('train process is over')
        self.Ypred = self.model.predict(self.Xtest)
        self.score = r2_score(y_true=self.Ytest, y_pred=self.Ypred)
        st.write('R2: {}'.format(self.score))

    def RandomForestRegressor(self):
        self.Ytest = self.Ytest.reset_index(drop=True)
        self.Xtest = self.Xtest.reset_index(drop=True)
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)
        # st.info('train process is over')
        self.Ypred = self.model.predict(self.Xtest)
        self.score = r2_score(y_true=self.Ytest,y_pred=self.Ypred)
        st.write('R2: {}'.format(self.score))

    def SupportVector(self):

        self.Ytest = self.Ytest.reset_index(drop=True)
        self.Xtest = self.Xtest.reset_index(drop=True)
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)
        # st.info('train process is over')
        self.Ypred = self.model.predict(self.Xtest)
        self.score = r2_score(y_true=self.Ytest,y_pred=self.Ypred)
        st.write('R2: {}'.format(self.score))
    
    def GPRegressor(self):

        self.Ytest = self.Ytest.reset_index(drop=True)
        self.Xtest = self.Xtest.reset_index(drop=True)
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)
        # st.info('train process is over')
        self.Ypred, self.Ystd = self.model.predict(self.Xtest)
        self.score = r2_score(y_true=self.Ytest,y_pred=self.Ypred)
        st.write('R2: {}'.format(self.score))

    def KNeighborsRegressor(self):

        self.Ytest = self.Ytest.reset_index(drop=True)
        self.Xtest = self.Xtest.reset_index(drop=True)
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)
        # st.info('train process is over')
        self.Ypred = self.model.predict(self.Xtest)
        self.score = r2_score(y_true=self.Ytest,y_pred=self.Ypred)
        st.write('R2: {}'.format(self.score))
   
    def LinearRegressor(self):
        self.Ytest = self.Ytest.reset_index(drop=True)
        self.Xtest = self.Xtest.reset_index(drop=True)
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        model = self.model.fit(self.Xtrain, self.Ytrain)
        # st.info('train process is over')
        self.Ypred = model.predict(self.Xtest)
        self.score = r2_score(y_true=self.Ytest,y_pred=self.Ypred)
        st.write('R2: {}'.format(self.score))
        truncate_func = np.vectorize(lambda x: '{:,.4f}'.format(x))
        coef = truncate_func(np.array(model.coef_)).reshape(1,-1)
        intercept = truncate_func(np.array(model.intercept_)).reshape(1,-1)
        st.write('coef: {}'.format(coef))
        st.write('intercept: {}'.format(intercept))
    def LassoRegressor(self):
        self.Ytest = self.Ytest.reset_index(drop=True)
        self.Xtest = self.Xtest.reset_index(drop=True)
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        model = self.model.fit(self.Xtrain, self.Ytrain)
        # st.info('train process is over')
        self.Ypred = model.predict(self.Xtest)
        self.score = r2_score(y_true=self.Ytest,y_pred=self.Ypred)
        st.write('R2: {}'.format(self.score))
        truncate_func = np.vectorize(lambda x: '{:,.4f}'.format(x))
        coef = truncate_func(np.array(model.coef_)).reshape(1,-1)
        intercept = truncate_func(np.array(model.intercept_)).reshape(1,-1)
        st.write('coef: {}'.format(coef))
        st.write('intercept: {}'.format(intercept))

    def RidgeRegressor(self):
        self.Ytest = self.Ytest.reset_index(drop=True)
        self.Xtest = self.Xtest.reset_index(drop=True)
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        model = self.model.fit(self.Xtrain, self.Ytrain)
        # st.info('train process is over')
        self.Ypred = model.predict(self.Xtest)
        self.score = r2_score(y_true=self.Ytest,y_pred=self.Ypred)
        st.write('R2: {}'.format(self.score))
        truncate_func = np.vectorize(lambda x: '{:,.4f}'.format(x))
        coef = truncate_func(np.array(model.coef_)).reshape(1,-1)
        intercept = truncate_func(np.array(model.intercept_)).reshape(1,-1)
        st.write('coef: {}'.format(coef))
        st.write('intercept: {}'.format(intercept))

    def MLPRegressor(self):
        self.Ytest = self.Ytest.reset_index(drop=True)
        self.Xtest = self.Xtest.reset_index(drop=True)
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)
        # st.info('train process is over')
        self.Ypred = self.model.predict(self.Xtest)
        self.score = r2_score(y_true=self.Ytest,y_pred=self.Ypred)
        st.write('R2: {}'.format(self.score))     

    def BaggingRegressor(self):
        self.Ytest = self.Ytest.reset_index(drop=True)
        self.Xtest = self.Xtest.reset_index(drop=True)
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)
        # st.info('train process is over')
        self.Ypred = self.model.predict(self.Xtest)
        self.score = r2_score(y_true=self.Ytest,y_pred=self.Ypred)
        st.write('R2: {}'.format(self.score))     

    def AdaBoostRegressor(self):
        self.Ytest = self.Ytest.reset_index(drop=True)
        self.Xtest = self.Xtest.reset_index(drop=True)
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)
        # st.info('train process is over')
        self.Ypred = self.model.predict(self.Xtest)
        self.score = r2_score(y_true=self.Ytest,y_pred=self.Ypred)
        st.write('R2: {}'.format(self.score)) 

    def GradientBoostingRegressor(self):
        self.Ytest = self.Ytest.reset_index(drop=True)
        self.Xtest = self.Xtest.reset_index(drop=True)
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)
        # st.info('train process is over')
        self.Ypred = self.model.predict(self.Xtest)
        self.score = r2_score(y_true=self.Ytest,y_pred=self.Ypred)
        st.write('R2: {}'.format(self.score))

    def XGBRegressor(self):
        self.Ytest = self.Ytest.reset_index(drop=True)
        self.Xtest = self.Xtest.reset_index(drop=True)
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)
        # st.info('train process is over')
        self.Ypred = self.model.predict(self.Xtest)
        self.score = r2_score(y_true=self.Ytest,y_pred=self.Ypred)
        st.write('R2: {}'.format(self.score))   
    
    def CatBRegressor(self):
        self.Ytest = self.Ytest.reset_index(drop=True)
        self.Xtest = self.Xtest.reset_index(drop=True)
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)
        # st.info('train process is over')
        self.Ypred = self.model.predict(self.Xtest)
        self.score = r2_score(y_true=self.Ytest,y_pred=self.Ypred)
        st.write('R2: {}'.format(self.score))
           
    def GPRegressor(self):
     
        self.Ytest = self.Ytest.reset_index(drop=True)
        self.Xtest = self.Xtest.reset_index(drop=True)
        self.Ytrain = self.Ytrain.astype('float')
        self.Ytest = self.Ytest.astype('float')

        self.model.fit(self.Xtrain, self.Ytrain)
        # st.info('train process is over')
        self.Ypred= self.model.predict(self.Xtest)
        self.score = r2_score(y_true=self.Ytest,y_pred=self.Ypred)
        st.write('R2: {}'.format(self.score))
        
    
class CLUSTER():

    def __init__(self, features, targets) -> None:
 
        self.features = features
        self.targets = targets   

        self.pred = None
        self.model = None

        self.score = None

    def K_means(self):

        self.model.fit(self.features)
    
        self.score = silhouette_score(self.features, self.model.labels_)
        # st.info('train process is over')
        st.write('silhouette score: {}'.format(self.score))
 

class SAMPLING:
    def __init__(self, features, targets) -> None:
    # origin features and targets
        self.features = features
        self.targets = targets   

        self.model = None

        self.score = None

        self.Ypred = None


def plot_and_export_results(model, model_name):
        plot = customPlot()
        plot.pred_vs_actual(model.Ytest, model.Ypred)
        result_data = pd.concat([model.Ytest, pd.DataFrame(model.Ypred)], axis=1)
        result_data.columns = ['actual', 'prediction']
        with st.expander("model"):
            tmp_download_link = download_button(model.model, model_name+'.pickle', button_text='download')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
        with st.expander('prediciton'):
            st.write(result_data)
            tmp_download_link = download_button(result_data, f'prediction.csv', button_text='download')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

def plot_and_export_results_clf(model, model_name, col_name, unique_categories):
    model.Ytest[col_name[0]] = pd.Series(unique_categories).iloc[model.Ytest[col_name[0]]].values
    model.Ypred[col_name[0]] = pd.Series(unique_categories).iloc[model.Ypred[col_name[0]]].values

    conf_matrix = confusion_matrix(model.Ytest.values, model.Ypred.values)
    conf_df = pd.DataFrame(conf_matrix, index=np.unique(model.Ytest), columns=np.unique(model.Ytest))
    with plt.style.context(['nature','no-latex']):
        fig, ax = plt.subplots()
        sns.heatmap(conf_df, annot=True, cmap='Blues',fmt="d",
                    xticklabels=conf_df.columns, yticklabels=conf_df.index)
        plt.xlabel('Actual')
        plt.ylabel('Prediction')
        plt.title('Confusion Matrix')
        st.pyplot(fig)
    with st.expander("model"):
        tmp_download_link = download_button(model.model, model_name+'_model.pickle', button_text='download')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
    result_data = pd.concat([pd.DataFrame(model.Ytest), pd.DataFrame(model.Ypred)], axis=1)
    result_data.columns = ['actual','prediction']
    with st.expander('Actual vs Predict'):
        st.write(result_data)
        tmp_download_link = download_button(result_data, f'actual vs prediction.csv', button_text='download')
        st.markdown(tmp_download_link, unsafe_allow_html=True)       

def export_cross_val_results(model, F, model_name, random_state):
    # chose the last mode in CV
    Y_pred, Y_test = Ffold_cross_val(model.features, model.targets, F, model.model, random_state)    
    st.write('R2: {}'.format(r2_score(y_true=Y_test, y_pred=Y_pred)))
    plot = customPlot()
    plot.pred_vs_actual(Y_test, Y_pred)                    
    with st.expander("model"):
        tmp_download_link = download_button(model.model, model_name+'.pickle', button_text='download')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
    result_data = pd.concat([pd.DataFrame(Y_test), pd.DataFrame(Y_pred)], axis=1)
    result_data.columns = ['actual', 'prediction']
    with st.expander('prediction'):
        st.write(result_data)
        tmp_download_link = download_button(result_data, f'prediction.csv', button_text='download')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

def cv_cal(model, F, random_state):
    # chose the last mode in CV
    Y_pred, Y_test = Ffold_cross_val(model.features, model.targets, F, model.model, random_state)    
    cv_score = r2_score(y_true=Y_test, y_pred=Y_pred)
    return cv_score

def export_cross_val_results_clf(model, F, model_name, col_name, unique_categories, random_state):
    
    Y_pred, Y_test = Ffold_cross_val(model.features, model.targets, F, model.model, random_state) 
    model.Ytest = pd.DataFrame(Y_test, columns=model.targets.columns)
    model.Ypred = pd.DataFrame(Y_pred, columns=model.targets.columns)

    st.write('accuracy score: {}'.format(accuracy_score(y_true=Y_test, y_pred=Y_pred)))

    model.Ytest[col_name[0]] = pd.Series(unique_categories).iloc[model.Ytest[col_name[0]]].values
    model.Ypred[col_name[0]] = pd.Series(unique_categories).iloc[model.Ypred[col_name[0]]].values
    conf_matrix = confusion_matrix(model.Ytest.values, model.Ypred.values, labels=np.unique(model.Ytest))
    conf_df = pd.DataFrame(conf_matrix, index=np.unique(model.Ytest), columns=np.unique(model.Ytest))
    with plt.style.context(['nature','no-latex']):
        fig, ax = plt.subplots()
        sns.heatmap(conf_df, annot=True, cmap='Blues',fmt="d",
                    xticklabels=conf_df.columns, yticklabels=conf_df.index)
        plt.xlabel('Actual')
        plt.ylabel('Prediction')
        plt.title('Confusion Matrix')
        st.pyplot(fig)

    with st.expander("model"):
        tmp_download_link = download_button(model.model, model_name+'.pickle', button_text='download')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

    result_data = pd.concat([model.Ytest, model.Ypred], axis=1)
    result_data.columns = ['actual','prediction']
    with st.expander('Actual vs Predict'):
        st.write(result_data)
        tmp_download_link = download_button(result_data, f'prediction.csv', button_text='download')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

def cv_cal_clf(model, F, random_state):
    Y_pred, Y_test = Ffold_cross_val(model.features, model.targets, F, model.model, random_state) 
    model.Ytest = pd.DataFrame(Y_test, columns=model.targets.columns)
    model.Ypred = pd.DataFrame(Y_pred, columns=model.targets.columns)
    cv_score = accuracy_score(y_true=Y_test, y_pred=Y_pred)
    return cv_score


def export_loo_results(model, loo, model_name):
    Y_pred  =[]
    Y_test = []
    features = pd.DataFrame(model.features).values
    targets = model.targets.values

    for train,test in loo.split(features):
        # Xtrain, Xtest, Ytrain,Ytest = features[train],features[test],targets[train],targets[test]
        Xtrain, Xtest = features[train], features[test]
        Ytrain, Ytest = targets[train], targets[test]
        model.model.fit(Xtrain, Ytrain)
        Ypred = model.model.predict(Xtest)
        Y_pred.append(Ypred)
        Y_test.append(Ytest)
    st.write('R2: {}'.format(r2_score(y_true=Y_test, y_pred=Y_pred)))
    plot = customPlot()
    plot.pred_vs_actual(Y_test, Y_pred)
    with st.expander("model"):
        tmp_download_link = download_button(model.model, model_name+'_model.pickle', button_text='download')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
    result_data = pd.concat([pd.DataFrame(Y_test), pd.DataFrame(Y_pred)], axis=1)
    result_data.columns = ['actual', 'prediction']
    with st.expander('prediciton'):
        st.write(result_data)
        tmp_download_link = download_button(result_data, f'prediction.csv', button_text='download')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
def loo_cal(model, loo):
    Y_pred  =[]
    Y_test = []
    features = pd.DataFrame(model.features).values
    targets = model.targets.values

    for train,test in loo.split(features):
        Xtrain, Xtest, Ytrain,Ytest = features[train],features[test],targets[train],targets[test]
        
        model.model.fit(Xtrain, Ytrain)
        Ypred = model.model.predict(Xtest)
        Y_pred.append(Ypred)
        Y_test.append(Ytest)
    loo_score = r2_score(y_true=Y_test, y_pred=Y_pred)
    return loo_score


def export_loo_results_clf(model, loo, model_name, col_name, unique_categories):
    Y_pred  =[]
    Y_test = []
    features = pd.DataFrame(model.features).values
    targets = model.targets.values

    for train,test in loo.split(features):
        # Xtrain, Xtest, Ytrain,Ytest = features[train],features[test],targets[train],targets[test]
        Xtrain, Xtest = features[train], features[test]
        Ytrain, Ytest = targets[train], targets[test]
        model.model.fit(Xtrain, Ytrain)
        Ypred = model.model.predict(Xtest)
        Y_pred.append(Ypred)
        Y_test.append(Ytest)
    Y_pred = np.array(Y_pred).reshape(-1,1)
    Y_test = np.array(Y_test).reshape(-1,1)
    model.Ypred = pd.DataFrame(Y_pred, columns=model.targets.columns) 
    model.Ytest = pd.DataFrame(Y_test, columns=model.targets.columns)        
    
    st.write('accuracy score: {}'.format(accuracy_score(y_true=Y_test, y_pred=Y_pred)))

    model.Ytest[col_name[0]] = pd.Series(unique_categories).iloc[model.Ytest[col_name[0]]].values
    model.Ypred[col_name[0]] = pd.Series(unique_categories).iloc[model.Ypred[col_name[0]]].values
    conf_matrix = confusion_matrix(model.Ytest.values, model.Ypred.values, labels=np.unique(model.Ytest))
    conf_df = pd.DataFrame(conf_matrix, index=np.unique(model.Ytest), columns=np.unique(model.Ytest))
    with plt.style.context(['nature','no-latex']):
        fig, ax = plt.subplots()
        sns.heatmap(conf_df, annot=True, cmap='Blues',fmt="d",
                    xticklabels=conf_df.columns, yticklabels=conf_df.index)
        plt.xlabel('Actual')
        plt.ylabel('Prediction')
        plt.title('Confusion Matrix')
        st.pyplot(fig)

    with st.expander("model"):
        tmp_download_link = download_button(model.model, model_name+'_model.pickle', button_text='download')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
    result_data = pd.concat([pd.DataFrame(Y_test), pd.DataFrame(Y_pred)], axis=1)
    result_data.columns = ['actual', 'prediction']
    with st.expander('prediciton'):
        st.write(result_data)
        tmp_download_link = download_button(result_data, f'prediction.csv', button_text='download')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
def loo_cal_clf(model, loo):
    Y_pred  =[]
    Y_test = []
    features = pd.DataFrame(model.features).values
    targets = model.targets.values

    for train,test in loo.split(features):
        Xtrain, Xtest, Ytrain,Ytest = features[train],features[test],targets[train],targets[test]
        
        model.model.fit(Xtrain, Ytrain)
        Ypred = model.model.predict(Xtest)
        Y_pred.append(Ypred)
        Y_test.append(Ytest)
    Y_pred = np.array(Y_pred).reshape(-1,1)
    Y_test = np.array(Y_test).reshape(-1,1)
    model.Ypred = pd.DataFrame(Y_pred, columns=model.targets.columns) 
    model.Ytest = pd.DataFrame(Y_test, columns=model.targets.columns)        
    loo_score = accuracy_score(y_true=Y_test, y_pred=Y_pred)
    return loo_score

def get_column_min(matrix):
    num_cols = len(matrix[0])
    result = []

    for col in range(num_cols):
        min_val = matrix[0][col]
        for row in range(1, len(matrix)):
            if matrix[row][col] < min_val:
                min_val = matrix[row][col]
        result.append(min_val)

    return result

def get_column_max(matrix):
    num_cols = len(matrix[0])
    result = []

    for col in range(num_cols):
        max_val = matrix[0][col]
        for row in range(1, len(matrix)):
            if matrix[row][col] > max_val:
                max_val = matrix[row][col]
        result.append(max_val)

    return result

def calculate_mean(list1, list2):
    result = [(x + y) / 2 for x, y in zip(list1, list2)]
    return result

def non_dominated_sorting(fitness_values):
    num_solutions = fitness_values.shape[0]

    domination_counts = np.zeros(num_solutions, dtype=int)
    dominated_solutions = [[] for _ in range(num_solutions)]
    frontiers = [[]]

    for i in range(num_solutions):
        for j in range(i + 1, num_solutions):
            if np.all(fitness_values[i] <= fitness_values[j]):
                if np.any(fitness_values[i] < fitness_values[j]):
                    domination_counts[j] += 1
                else:
                    dominated_solutions[i].append(j)
            elif np.all(fitness_values[i] >= fitness_values[j]):
                if np.any(fitness_values[i] > fitness_values[j]):
                    domination_counts[i] += 1
                else:
                    dominated_solutions[j].append(i)

        if domination_counts[i] == 0:
            frontiers[0].append(i)

    i = 0
    while len(frontiers[i]) > 0:
        next_frontier = []
        for j in frontiers[i]:
            for k in dominated_solutions[j]:
                domination_counts[k] -= 1
                if domination_counts[k] == 0:
                    next_frontier.append(k)
        i += 1
        frontiers.append(next_frontier)

    return frontiers[:-1]

def find_non_dominated_solutions(fitness_values, feature_name):
    fronts = non_dominated_sorting(fitness_values)
    non_dominated_solutions = []
    for front in fronts:
        non_dominated_solutions.extend(front)
    non_dominated_solutions_Data = fitness_values[non_dominated_solutions]
    non_dominated_solutions_Data = pd.DataFrame(non_dominated_solutions_Data, columns=feature_name)
    non_dominated_solutions_Data.sort_values(by=feature_name[0], inplace=True)
    return  non_dominated_solutions_Data

def dominated_hypervolume(pareto_data, ref_point):
    pareto_data = np.vstack([pareto_data, ref_point])
    pareto_data = pareto_data[np.argsort(-pareto_data[:,0])]
    S = 0
    for i in range(pareto_data.shape[0]-1):
        S += (pareto_data[i,0] - pareto_data[i+1,0]) * (pareto_data[0,1] - pareto_data[i+1,1])
    return S


def Ffold_cross_val(Xtrain, Ytrain, F, estimator, random_state):
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    row_Xtrain = Xtrain.shape[0]
    kf = KFold(n_splits=F, shuffle=True, random_state=random_state)
    predict = np.zeros([row_Xtrain, 1])
    real = np.zeros([row_Xtrain, 1])
    for train_index, val_index in kf.split(Xtrain):
        x_train, x_val = [Xtrain[i] for i in train_index], [Xtrain[i] for i in val_index]
        y_train, y_val = [Ytrain[i] for i in train_index], [Ytrain[i] for i in val_index]
        estimator.fit(x_train, y_train)
        predict[val_index] = estimator.predict(x_val).reshape(-1,1)
        real[val_index] = np.array(y_val).reshape(-1,1)
    return predict, real

def normalize(data, normalize: Optional[str]=None):

    columns = data.columns
    X = data.values
    
    if normalize == 'StandardScaler':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif normalize == 'MinMaxScaler':
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    X = pd.DataFrame(X, columns=columns)
    return X, scaler

def inverse_normalize(data, scaler, normalize: Optional[str]=None):
    columns = data.columns
    X = data.values
    if normalize == 'StandardScaler':
        X = scaler.inverse_transform(data)

    elif normalize == 'MinMaxScaler':
        X = scaler.inverse_transform(data)
    X = pd.DataFrame(X, columns=columns)        
    return X


def pca_inverse_normalize(data, scaler, normalize: Optional[str]=None):
    if normalize == 'StandardScaler':
        data = scaler.inverse_transform(data)
    elif normalize == 'MinMaxScaler':
        data = scaler.inverse_transform(data)       
    return data

def feature_transform(df, option):
    if option == 'Alloy':
        df = StrToComposition().featurize_dataframe(df, "Alloy")
        HEA = WenAlloys()
        df = HEA.featurize_dataframe(df, col_id="composition", ignore_errors=True)  # input the "composition" column to the featurizer
    elif option == 'Inorganic':
        df = StrToComposition().featurize_dataframe(df, "Inorganic")
        EP = ElementProperty.from_preset(preset_name="magpie")
        df = EP.featurize_dataframe(df, col_id="composition", ignore_errors=True)    
    else:
        st.warning('Table title need be named "Alloy" or "Inorganic"')     
    return df

