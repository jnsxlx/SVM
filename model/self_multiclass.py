import numpy as np
from sklearn import svm


class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        W = np.zeros((K, d))

        n_iter = 1500
        learning_rate = 1e-8
        for i in range(n_iter):
            W -= learning_rate * self.grad_student(W, X_intercept, y)

        self.W = W

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return np.argmax(self.W.dot(X_intercept.T), axis=0)

    def bsvm_ovr_student(self, X, y):
        '''
        Train OVR binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        '''
        binary_svm = {}
        labels = np.unique(y)
        for i in labels:
            y_bin = np.ones(len(y))
            y_reshape = y.reshape(y_bin.shape)
            y_bin[y_reshape != i] = 0
            y_bin[y_reshape == i] = 1

            clf = svm.LinearSVC(random_state=12345)
            clf.fit(X, y_bin)
            binary_svm[i] = clf
        return binary_svm



    def bsvm_ovo_student(self, X, y):
        '''
        Train OVO binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        '''
        binary_svm = {}
        labels = np.unique(y)
        n = len(labels)
        for i in range(n):
            for j in range(i + 1, n):
                data = np.hstack([X, y.reshape(-1, 1)])
                data_0 = data[data[:, -1] == labels[i]]
                data_0[:, -1] = 0
                data_1 = data[data[:, -1] == labels[j]]
                data_1[:, -1] = 1
                data = np.vstack([data_0, data_1])

                clf = svm.LinearSVC(random_state=12345)
                clf.fit(data[:, 0:-1], data[:, -1])
                binary_svm[(labels[i], labels[j])] = clf

        return binary_svm

    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        labels = np.array(list(self.binary_svm.keys()))
        n = len(labels)
        N_sp = X.shape[0]
        scores = np.ones((N_sp, n))

        for i in sorted(labels):
            clf = self.binary_svm[i]
            scores[:, i] = clf.decision_function(X)

        return scores

    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        labels = np.unique(np.array(list(self.binary_svm.keys())))
        n = len(labels)
        N_sp = X.shape[0]
        scores = np.ones((N_sp, n))

        for i, clf in self.binary_svm.items():
            y_est = clf.predict(X)
            scores[:, i[0]] = scores[:, i[0]] + (y_est == 0)
            scores[:, i[1]] = scores[:, i[1]] + (y_est == 1)

        return scores

    def loss_student(self, W, X, y, C=1.0):
        '''
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        '''
        loss_r = 0.5 * np.dot(W.flatten(), W.flatten().T)
        loss_l = 0
        N_sp = X.shape[0]

        for i in range(N_sp):
            prod = 1 + np.dot(W, X[i, :].reshape(-1, 1))
            prod[y[i]] -= 1
            loss_l += np.max(prod)
            loss_l -= np.dot(W[y[i], :].reshape(1, -1), X[i, :].reshape(-1, 1))

        loss = (loss_r + C * loss_l).flatten()

        return loss

    def grad_student(self, W, X, y, C=1.0):
        '''
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The graident of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        '''
        loss_l = np.zeros(W.shape)
        N_sp = X.shape[0]

        for i in range(N_sp):
            prod = 1 + np.dot(W, X[i, :].reshape(-1, 1))
            prod[y[i]] -= 1
            prod_max = np.argmax(prod)
            loss_l[prod_max, :] += X[i, :]
            loss_l[y[i], :] -= X[i, :]

        grad = W + C * loss_l

        return grad
