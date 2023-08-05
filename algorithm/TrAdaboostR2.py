import numpy as np

class TrAdaboostR2:
    def __init__(self, estimator, N) -> None:
        self.estimator = estimator
        self.N = N
        self.estimators = []
        self.beta_N = np.zeros(self.N)

    def fit(self, Xsource, Xtarget, Ysource, Ytarget):
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
        results = np.ones([n + m, self.N])
        error_rates = []
        for i in range(self.N):
            P = self._calculate_weight(weights)
            self.estimator.fit(Xtrain, Ytrain, sample_weight=P[:, 0])
            self.estimators.append(self.estimator)
            Ypred = self.estimator.predict(Xtrain)
            error_rate = self._calculate_error_rate(Ypred[n:], Ytarget, P[n:, :])
            error_rates.append(error_rate)
            print(f"Iteration {i+1} / {self.N} | error rate in target data {error_rate:.3f}")
            if error_rate <= 1e-10 or error_rate > 0.5:
                print(f"Early stopping at {i} iteration ...")
                break
            self.beta_N[i] = error_rate / (1 - error_rate)
            # adjust the sample weight
            Z_t = np.abs(np.array(Ypred - Ytrain)).max()
            for t in range(m):
                weights[n + t] = weights[t] * np.power(self.beta_N[i], -np.abs(Ypred[n+t] - Ytrain[n+t])) / Z_t
            for s in range(n):
                weights[s] = weights[s] * np.power(beta, np.abs(Ypred[s] - Ytrain[s])) / Z_t

    def _calculate_weight(self, weights):
        sum_weight = np.sum(weights)
        return np.asarray(weights / sum_weight, order='C')

    def _calculate_error_rate(self, Ypred, Ytarget, weight_target):
        max_weight = np.abs(Ypred - Ytarget).max()
        # misclassify_num = np.sum(np.abs(Ypred - Ytarget))
        return np.sum(weight_target[:, 0] / max_weight * np.abs(Ypred - Ytarget))

    def predict(self, Xtest):
        Xtest = np.asarray(Xtest, order='C')
        result = np.ones([Xtest.shape[0], self.N])
        predict = []
        i = 0
        for estimator in self.estimators:
            Ypred = estimator.predict(Xtest)
            result[:, i] = Ypred
            i += 1
        for i in range(Xtest.shape[0]):
            predict.append(np.sum(result[i, int(np.ceil(self.N/2)) : self.N]) / (self.N - int(np.ceil(self.N/2))))
        return predict