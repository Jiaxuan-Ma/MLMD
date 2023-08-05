

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from tabulate import tabulate

import pandas as pd
from typing import Optional
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class mobo:
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
    def fit(self, X, y, visual_data, method, number, objective, ref_point):
        Xtrain = X
        Ytrain = y
        Xtest = visual_data 
        target_name = Ytrain.columns.tolist()
        feature_name = Xtrain.columns.tolist()
        if objective == 'min':
            kernel = RBF(length_scale=1.0)
            gp_model = GaussianProcessRegressor(kernel=kernel)
            gp_model.fit(Xtrain, Ytrain)
            Ypred, Ystd = gp_model.predict(Xtest, return_std=True)
            Ypred = pd.DataFrame(Ypred, columns=Ytrain.columns.tolist())

            if method == 'HV':
                    HV_values = []
                    for i in range(Ypred.shape[0]):
                        i_Ypred = Ypred.iloc[i]
                        Ytrain_i_Ypred = Ytrain.append(i_Ypred)
                        i_pareto_front = self.find_non_dominated_solutions(Ytrain_i_Ypred.values, Ytrain_i_Ypred.columns.tolist())
                        i_HV_value = self.dominated_hypervolume(i_pareto_front, ref_point)
                        HV_values.append(i_HV_value)
                    
                    HV_values = pd.DataFrame(HV_values, columns=['HV values'])
                    HV_values.set_index(Xtest.index, inplace=True)

                    max_idx = HV_values.nlargest(number, 'HV values').index
                    recommend_point = Xtest.loc[max_idx]
                    # Xtest = Xtest.drop(max_idx)
                    print('The maximum value of HV: \n ', tabulate(HV_values.loc[max_idx].values))
                    print('The recommended point is :\n', tabulate(recommend_point.values, headers = feature_name+target_name, tablefmt = 'pretty'))
            elif method == 'EHVI':
                pass
        elif objective == 'max':
            pass
        return HV, recommend_point

    def preprocess(self, data, target_number, normalize: Optional[str]=None):
        df = pd.read_csv(data)
        X = df.iloc[:,:-target_number].values
        y = df.iloc[:,-target_number:]
        if normalize == 'StandardScaler':
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif normalize == 'MinMaxScaler':
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        else:
            X = X        
        return X, y

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
    

# df = pd.read_csv('./test.csv')
# Xtrain = df.iloc[:,:-2]
# Ytrain = df.iloc[:,-2:]

# mobo = mobo ()

# mobo.fit(X = Xtrain, y = Ytrain, visual_data=Xtrain, method='HV',number= 1, objective='min', ref_point=[10, 10])
