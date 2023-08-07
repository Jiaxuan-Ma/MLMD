import numpy as np
from sklearn.tree import DecisionTreeRegressor
import streamlit as st

class TwoStageTrAdaboostR2:
    def __init__(self) -> None:
        # self.estimator = estimator
        # self.N = N
        self.estimators = []
        # predictions = []
    def fit(self, params, Xsource, Xtarget, Ysource, Ytarget, N):
        self.N = N
        self.beta_N = np.zeros(self.N)
        Xsource = np.asarray(Xsource, order='C')
        Xtarget = np.asarray(Xtarget, order='C')
        Ysource = np.asarray(Ysource, order='C')
        Ytarget = np.asarray(Ytarget, order='C')
        Xtrain = np.concatenate([Xsource, Xtarget], axis=0)
        Ytrain = np.concatenate([Ysource, Ytarget], axis=0)
        n = Xsource.shape[0]
        m = Xtarget.shape[0]
        # initial weight
        weight_source = np.ones([n, 1]) / n
        weight_target = np.ones([m, 1]) / m
        weights = np.concatenate([weight_source, weight_target], axis=0)
        beta = 1 / (1 + np.sqrt(2 * np.log(n/self.N)))
        # log result
        # results = np.ones([n + m, self.N])
        error_rates = []
        for i in range(self.N):
            weights = self._calculate_weight(weights) # update weights
            estimator = DecisionTreeRegressor(random_state=params['random state'],splitter=params['splitter'],
                        max_depth=params['max depth'],min_samples_leaf=params['min samples leaf'],
                        min_samples_split=params['min samples split']) 
            estimator.fit(Xtrain, Ytrain, sample_weight=weights[:, 0])
            self.estimators.append(estimator)
            Ypred = estimator.predict(Xtrain)
            error_rate = self._calculate_error_rate(Ypred[n:], Ytarget, weights[n:, :])
            error_rates.append(error_rate)
            if error_rate <= 1e-10 or error_rate > 0.5:
                print(f"Early stopping at {i} iteration ...")
                self.N = i
                break
            print(f"Iteration {i+1} / {self.N} | error rate in target data {error_rate:.3f}")
            self.beta_N[i] = error_rate / (1 - error_rate)
            # adjust the sample weight
            Z_t = np.abs(np.array(Ypred - Ytrain)).max()
            for t in range(m):
                weights[n + t] = weights[n+t] * np.power(self.beta_N[i], -np.abs(Ypred[n+t] - Ytrain[n+t]) / Z_t)

            for s in range(n):
                weights[s] = weights[s] * np.power(beta, np.abs(Ypred[s] - Ytrain[s]) / Z_t)
        # for j in range(row_A):
        #     weights[j] = weights[j] * np.power(bata, np.abs(result_response[j, i] - response_A[j]) / D_t)
    def _calculate_weight(self, weights):
        sum_weight = np.sum(weights)
        return np.asarray(weights / sum_weight, order='C')

    def _calculate_error_rate(self, Ypred, Ytarget, weight_target):
        max_weight = np.abs(Ypred - Ytarget).max()
        # misclassify_num = np.sum(np.abs(Ypred - Ytarget))
        return np.sum(weight_target[:, 0] / max_weight * np.abs(Ypred - Ytarget))

    def estimators_predict(self, Xtest):
        
        Xtest = np.asarray(Xtest, order='C')
        self.predictions = np.ones([Xtest.shape[0], self.N])
        st.write(self.predictions.shape)
        for i, estimator in zip(range(self.N), self.estimators):
            Ypred = estimator.predict(Xtest)
            self.predictions[:, i] = Ypred
        predict = []
        for i in range(Xtest.shape[0]):
            predict.append(np.sum(self.predictions[i, int(np.ceil(self.N/2)):self.N]) / (self.N - int(np.ceil(self.N/2))))
        return predict

    # def predict_weighted_median(self):
    # sorted_predictions = sorted(zip(self.predictions, self.beta_N), reverse=True)
    # total_weight = sum(self.beta_N)
    # half_weight = np.ceil(total_weight / 2.0)
    # cumulative_weight = 0.0
    # median_predictions = []
    # for prediction, weight in sorted_predictions:
    #     cumulative_weight += weight
    #     median_predictions.append(prediction)
    #     if cumulative_weight >= half_weight:
    #         break
    #     return median_predictions[-1]
    # # for i in range(row_T):
    # #     predict[i] = np.sum(
    # #         result_response[row_A + row_S + i, int(np.floor(N / 2)):N]) / (N - int(np.floor(N / 2)))



