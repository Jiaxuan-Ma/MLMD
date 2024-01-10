
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from tabulate import tabulate

import pandas as pd
from typing import Optional
import numpy as np
import warnings

from scipy.stats import norm
import streamlit as st

warnings.filterwarnings('ignore')


class Mobo4mat:
    """
    Multi Objective Bayesian Optimization.

    Parameters
    ----------
    mission

    Attributes
    ----------

    Examples
    --------
    """
    def fit(self, X, y, visual_data, method, kernel_option, number, objective, ref_point):
        if objective == 'max':
            Xtrain = -X
            Ytrain = -y
            Xtest = -visual_data
            target_name = Ytrain.columns.tolist()
            feature_name = Xtrain.columns.tolist()
            ref_point = -np.array(ref_point)

        elif objective == 'min':
            Xtrain = X
            Ytrain = y
            Xtest = visual_data 
            target_name = Ytrain.columns.tolist()
            feature_name = Xtrain.columns.tolist()
            ref_point = np.array(ref_point)
        if kernel_option == 'rbf':
            kernel = RBF()
        elif kernel_option == 'DotProduct + WhiteKernel':
            kernel = DotProduct() + WhiteKernel()
        gp_model = GaussianProcessRegressor(kernel=kernel)
        gp_model.fit(Xtrain, Ytrain)

        Ypred, Ystd = gp_model.predict(Xtest, return_std=True)
        
        Ypred = pd.DataFrame(Ypred, columns=Ytrain.columns.tolist())
        Ystd = pd.DataFrame(Ystd, columns=['std1', 'std2'])

        if method == 'HV':
            HV_values = []
            for i in range(Ypred.shape[0]):
                i_Ypred = Ypred.iloc[i]
                Ytrain_i_Ypred = Ytrain._append(i_Ypred)
                i_pareto_front = self.find_non_dominated_solutions(Ytrain_i_Ypred.values, Ytrain_i_Ypred.columns.tolist())
                i_HV_value = self.dominated_hypervolume(i_pareto_front, ref_point)
                HV_values.append(i_HV_value)
            
            HV_values = pd.DataFrame(HV_values, columns=['HV values'])
            HV_values.set_index(Xtest.index, inplace=True)

            max_idx = HV_values.nlargest(number, 'HV values').index
            
            Ypred_recommend = Ypred.iloc[max_idx]
            if objective == 'max':
                Ypred_recommend = - Ypred_recommend
                recommend_point = -Xtest.loc[max_idx]
            elif objective == 'min':        
                recommend_point = Xtest.loc[max_idx]

            print('The maximum value of HV: \n ', tabulate(HV_values.loc[max_idx].values))
            print('The recommended value is : \n ', tabulate(Ypred_recommend))
            print('The recommended point is :\n', tabulate(recommend_point.values, headers = feature_name+target_name, tablefmt = 'pretty'))

        if method == 'EHVI':
            HV_values = []
            Ypred_std = pd.concat([Ypred, Ystd],axis=1)
            for i in range(Ypred_std.shape[0]):
                i_Ypred_std = Ypred_std.iloc[i]
                i_Ypred = Ypred.iloc[i]
                Ytrain_i_Ypred = Ytrain.append(i_Ypred)
                i_pareto_front = self.find_non_dominated_solutions(Ytrain_i_Ypred.values, Ytrain_i_Ypred.columns.tolist())

                y_0 = pd.DataFrame([[ref_point[0], -np.inf]], columns=Ytrain.columns.tolist())
               

                i_EHVI_value = self.cal_EHVI(ref_point,i_pareto_front,i_Ypred_std, Ytrain.columns.tolist())
                HV_values.append(i_EHVI_value)

            HV_values = pd.DataFrame(HV_values, columns=['HV values'])
            HV_values.set_index(Xtest.index, inplace=True)

            max_idx = HV_values.nlargest(number, 'HV values').index
            
            Ypred_recommend = Ypred.iloc[max_idx]
            if objective == 'max':
                Ypred_recommend = - Ypred_recommend
                recommend_point = -Xtest.loc[max_idx]
            elif objective == 'min':        
                recommend_point = Xtest.loc[max_idx]

            print('The maximum value of HV: \n ', tabulate(HV_values.loc[max_idx].values))
            print('The recommended value is : \n ', tabulate(Ypred_recommend))
            print('The recommended point is :\n', tabulate(recommend_point.values, headers = feature_name+target_name, tablefmt = 'pretty'))            


        return HV_values.loc[max_idx].values, recommend_point, Ypred_recommend

    def func_psi(self, a, b, mean, std):
        norm_value = (b - mean) / std
        z = norm_value
        psi_ab = std * norm.pdf(z) + (a - mean) * norm.cdf(z)
        return psi_ab

    def cal_EHVI(self, ref_point, pareto_data, Ypred_std, target_name):
        pareto_data = pareto_data.sort_values(by=[target_name[0]], ascending=False)
        y_0 = pd.DataFrame([[ref_point[0], -np.inf]], columns=target_name)
        y_inf = pd.DataFrame([[-np.inf, ref_point[1]]], columns=target_name)
        pareto_data = pd.concat([y_0, pareto_data, y_inf], ignore_index=True)
        row_size= pareto_data.shape[0]-1
        EHVI = np.zeros(row_size)
        for i in range(row_size):
            z = (pareto_data.iloc[i+1][target_name[0]] - Ypred_std[target_name[0]]) / Ypred_std['std1']
            value_1 = (pareto_data.iloc[i][target_name[0]] - pareto_data.iloc[i+1][target_name[0]]) * norm.cdf(z) * \
                    self.func_psi(pareto_data.iloc[i+1][target_name[1]], pareto_data.iloc[i+1][target_name[1]], Ypred_std[target_name[1]], Ypred_std['std2'])
            value_2 = (self.func_psi(pareto_data.iloc[i][target_name[0]], pareto_data.iloc[i][target_name[0]], Ypred_std[target_name[0]], Ypred_std['std1'])\
                        - self.func_psi(pareto_data.iloc[i][target_name[0]], pareto_data.iloc[i+1][target_name[0]], Ypred_std[target_name[0]], Ypred_std['std1'])) * \
                            self.func_psi(pareto_data.iloc[i+1][target_name[1]], pareto_data.iloc[i+1][target_name[1]], Ypred_std[target_name[1]], Ypred_std['std2'])
            EHVI[i] = value_1 + value_2
            
        EHVI = EHVI[~np.isnan(EHVI)]
        EHVI_sum = np.sum(EHVI)
        return EHVI_sum

    def non_dominated_sorting(self, fitness_values): # min
        num_solutions = fitness_values.shape[0]
        domination_counts = np.zeros(num_solutions, dtype=int)  
        dominated_solutions = [[] for _ in range(num_solutions)]  
        frontiers = []  

        for i in range(num_solutions): 
            for j in range(i + 1, num_solutions):  
                if np.all(fitness_values[i,:] <= fitness_values[j,:]): 
                    if np.any(fitness_values[i,:] < fitness_values[j,:]):  
                        domination_counts[j] += 1  
                    else:
                        dominated_solutions[i].append(j)  
                elif np.all(fitness_values[i,:] >= fitness_values[j,:]):  
                    if np.any(fitness_values[i,:] > fitness_values[j,:]):  
                        domination_counts[i] += 1 
                    else:
                        dominated_solutions[j].append(i) 

            if domination_counts[i] == 0: 
                frontiers.append(i)  

                i = 0
                while i < len(frontiers):  
                    current_frontier = frontiers[i]
                    next_frontier = []
                    for j in dominated_solutions[current_frontier]:  
                        domination_counts[j] -= 1  
                        if domination_counts[j] == 0:  
                            next_frontier.append(j)  
                    i += 1
                    frontiers.extend(next_frontier)  
        return frontiers
    
    def find_non_dominated_solutions(self, fitness_values, feature_name):
        frontiers = self.non_dominated_sorting(fitness_values)

        non_dominated_solutions_Data = fitness_values[frontiers]
        non_dominated_solutions_Data = pd.DataFrame(non_dominated_solutions_Data, columns=feature_name)
        non_dominated_solutions_Data.sort_values(by=feature_name[0], inplace=True)

        return  non_dominated_solutions_Data

    def dominated_hypervolume(self, pareto_data, ref_point):
        pareto_data = np.vstack([pareto_data, ref_point])
        pareto_data = pareto_data[np.argsort(-pareto_data[:,0])]
        S = 0
        for i in range(pareto_data.shape[0]-1):
            S += (pareto_data[i,0] - pareto_data[i+1,0]) * (pareto_data[0,1] - pareto_data[i+1,1])
        return S
    
    def func_selector(selector, Ytrain, Ypred, Ystd):
        if selector == 'EGO':
            z = (np.min(Ytrain) -  Ypred) / Ystd
            ego = Ystd * z * norm.cdf(z) + Ystd * norm.pdf(z)
            value = pd.DataFrame(ego, columns=['EGO'])
        elif selector == 'PI':
            z = (np.min(Ytrain) -  Ypred) / Ystd
            pi = z * norm.cdf(z) + norm.pdf(z)
            value = pd.DataFrame(pi, columns=['PI'])
        elif selector == 'UCB':
            para = 0.5
            ucb = Ypred - Ystd*para
            value = pd.DataFrame(ucb, columns=['UCB'])
        
        value = round(value, 5)
        return value

