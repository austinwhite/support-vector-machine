import numpy as np

class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        _, n_feature = X.shape

        self.w = np.zeros(n_feature)
        self.b = 0
        
        for _ in range(self.n_iters):
            for i, x_i in enumerate(X):
                condition = y_[i] * (np.dot(x_i, self.w) - self.b)
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[i]))
                    self.b -= self.lr * y_[i]

    def predict(self, X):
        linear_out = np.dot(X, self.w) - self.b
        return np.sign(linear_out)
