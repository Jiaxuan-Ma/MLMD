import numpy as np
from sklearn.tree import DecisionTreeRegressor

class TrAdaboostR2:
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
            if Z_t == 0: Z_t = 1e-5
            for t in range(m):
                weights[n + t] = weights[n+t] * np.power(self.beta_N[i], -np.abs(Ypred[n+t] - Ytrain[n+t]) / Z_t)
            for s in range(n):
                weights[s] = weights[s] * np.power(beta, np.abs(Ypred[s] - Ytrain[s]) / Z_t)

    def _calculate_weight(self, weights):
        sum_weight = np.sum(weights)
        return np.asarray(weights / sum_weight, order='C')

    def _calculate_error_rate(self, Ypred, Ytarget, weight_target):
        max_weight = np.abs(Ypred - Ytarget).max()

        return np.sum(weight_target[:, 0] / max_weight * np.abs(Ypred - Ytarget))

    def predict(self, Xtest):
        Xtest = np.asarray(Xtest, order='C')
        self.estimators_predicts = np.ones([Xtest.shape[0], self.N])
        for i, estimator in zip(range(self.N), self.estimators):
            Ypred = estimator.predict(Xtest)
            self.estimators_predicts[:, i] = Ypred
        
        predicts = self.beta_N[:self.N] + self.estimators_predicts
        predicts = predicts[:, int(np.ceil(self.N/2)):self.N] 
        median_predicts = []
        self.best_estimators = []
        for i in range(predicts.shape[0]):
            # row_median = np.median(predicts[i, :])  # median value
            row_idx = np.argsort(predicts[i, :])  # median index
            median_idx = len(row_idx) // 2 
            median_predicts.append(self.estimators_predicts[i, int(np.ceil(self.N/2))+row_idx[median_idx]])
            # log estimators
            self.best_estimators.append(self.estimators[int(np.ceil(self.N/2))+row_idx[median_idx]])
        # st.write(self.best_estimators)
        return median_predicts

    
