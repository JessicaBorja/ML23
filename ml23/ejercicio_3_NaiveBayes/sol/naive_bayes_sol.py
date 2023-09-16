import numpy as np

class NaiveBayes():
    def __init__(self, alpha=1) -> None:
        self.alpha = alpha


    def fit(self, X, y):
        # TODO: Calcula la probabilidad de que una muestra sea positiva P(y=1)
        self.prior_positives = np.sum(y) / len(y)

        # TODO: Calcula la probabilidad de que una muestra sea negativa P(y=0)
        self.prior_negative = 1 - self.prior_positives

        # TODO: Para cada palabra del vocabulario x_i
        # calcula la probabilidad de: P(x_i| y=1)
        # Guardalas en un arreglo de numpy:
        # self._likelihoods_positives = [P(x_1| y=1), P(x_2| y=1), ..., P(x_n| y=1)]

        _positives_with_word = np.sum(X[y==1], axis=0)
        _total_positives = np.sum(y==1)
        _likelihoods = self.alpha + _positives_with_word
        _likelihoods /= 1 + self.alpha + _total_positives
        self._likelihoods_positives = _likelihoods
        
        # P(x_i| y=0)
        _negatives_with_word = np.sum(X[y==0], axis=0)
        _total_negatives = np.sum(y==0)
        _likelihoods_negatives = self.alpha + _negatives_with_word 
        _likelihoods_negatives /= 1 + self.alpha + _total_negatives
        self._likelihoods_negatives = _likelihoods_negatives
        return self

    def predict(self, X):
        # P(y = 1 | X)
        posterior = self.prior_positives # P(y = 1)
        probabilities = np.where(X, self._likelihoods_positives, 1 - self._likelihoods_positives)
        posterior *= np.prod(probabilities, axis=1) # P(y) * P(x1|y) * P(x2|y) * ... * P(xn|y)

        # P(y = 0 | X)
        posterior_negative = 1 - self.prior_positives # P(y = 0)
        probabilities = np.where(X, self._likelihoods_negatives, 1 - self._likelihoods_negatives)
        posterior_negative *= np.prod(probabilities, axis=1)

        return posterior > posterior_negative
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)