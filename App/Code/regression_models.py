import numpy as np
from sklearn.model_selection import KFold


class LinearRegression(object):

    kfold = KFold(n_splits=5)

    def __init__(self, regularization, lr=0.001, method='batch', num_epochs=500, bs=50,
                 cv=kfold, initialise_method='zeros', use_momentum=False, momentum=0.9):
        self.lr = lr
        self.num_epochs = num_epochs
        self.bs = bs
        self.method = method
        self.cv = cv
        self.regularization = regularization
        self.initialise_method = initialise_method
        self.use_momentum = use_momentum
        self.momentum = momentum

    def mse(self, ytrue, ypred):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]

    def r2(self, ytrue, ypred):
        ss_res = ((ytrue - ypred) ** 2).sum()
        ss_tot = ((ytrue - ytrue.mean()) ** 2).sum()
        return 1 - (ss_res / ss_tot)

    def weight_initialise(self, n_features):
        if self.initialise_method == "xavier":
            lower, upper = -(1.0 / np.sqrt(n_features)), (1.0 / np.sqrt(n_features))
            return np.random.uniform(lower, upper, size=n_features)
        else:
            return np.zeros(n_features)

    def fit(self, X_train, y_train):

        self.kfold_scores = list()
        self.kfold_r2_scores = list()

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):

            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val = X_train[val_idx]
            y_cross_val = y_train[val_idx]

            self.theta = self.weight_initialise(X_cross_train.shape[1])
            self.prev_step = np.zeros(X_cross_train.shape[1])
            self.val_loss_old = np.inf   # FIX: reset per fold, not once before all folds

            for epoch in range(self.num_epochs):

                perm = np.random.permutation(X_cross_train.shape[0])
                X_cross_train = X_cross_train[perm]
                y_cross_train = y_cross_train[perm]

                if self.method == 'sto':
                    for i in range(X_cross_train.shape[0]):
                        X_method_train = X_cross_train[i:i + 1, :]
                        y_method_train = y_cross_train[i:i + 1]
                        train_loss = self._train(X_method_train, y_method_train)

                elif self.method == 'mini':
                    for batch_idx in range(0, X_cross_train.shape[0], self.bs):
                        X_method_train = X_cross_train[batch_idx:batch_idx + self.bs, :]
                        y_method_train = y_cross_train[batch_idx:batch_idx + self.bs]
                        train_loss = self._train(X_method_train, y_method_train)

                else:
                    X_method_train = X_cross_train
                    y_method_train = y_cross_train
                    train_loss = self._train(X_method_train, y_method_train)

                yhat_val = self.predict(X_cross_val)
                val_loss_new = self.mse(y_cross_val, yhat_val)
                val_r2 = self.r2(y_cross_val, yhat_val)

                if np.allclose(val_loss_new, self.val_loss_old):
                    break
                self.val_loss_old = val_loss_new

            self.kfold_scores.append(val_loss_new)
            self.kfold_r2_scores.append(val_r2)

            print(f"Fold {fold}: MSE={val_loss_new}, R2={val_r2}")

    def _train(self, X, y):
        yhat = self.predict(X)
        m = X.shape[0]


        reg_grad = np.zeros_like(self.theta) + self.regularization.derivation(self.theta)
        reg_grad[0] = 0

        grad = (1 / m) * X.T @ (yhat - y) + reg_grad

        step = self.lr * grad
        if self.use_momentum:
            self.theta = self.theta - step + self.momentum * self.prev_step
        else:
            self.theta = self.theta - step
        self.prev_step = step

        return self.mse(y, yhat)

    def predict(self, X):
        return X @ self.theta

    def _coef(self):
        return self.theta[1:]

    def _bias(self):
        return self.theta[0]

    def plot_feature_importance(self, feature_names=None, top_n=None):
        import matplotlib.pyplot as plt

        coefs = self._coef()
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(coefs))]

        importance = np.abs(coefs)
        order = np.argsort(importance)

        if top_n is not None:
            order = order[-top_n:]

        plt.figure(figsize=(8, max(4, len(order) * 0.3)))
        plt.barh(np.array(feature_names)[order], importance[order])
        plt.xlabel('|coefficient|')
        plt.title('Feature importance (by coefficient magnitude)')
        plt.tight_layout()
        plt.show()


class Lasso:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(np.abs(theta))

    def derivation(self, theta):
        return self.l * np.sign(theta)


class Ridge:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(np.square(theta))

    def derivation(self, theta):
        return self.l * 2 * theta


class NoRegularization:
    def __init__(self, l=0):
        self.l = l

    def __call__(self, theta):
        return 0

    def derivation(self, theta):
        return 0
