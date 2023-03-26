import streamlit as st
from streamlit_shap import st_shap

import numpy as np
import pandas as pd


import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


import shap
import xgboost
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

import collections
import utils
import os
import sys
import gc
# utilities
from itertools import chain

import base64
import json
import pickle
import uuid
import re