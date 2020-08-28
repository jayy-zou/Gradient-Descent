import numpy as np


class Loss:
    def __init__(self, regularization=None):
        self.regularization = regularization

    def forward(self, X, w, y):
        pass

    def backward(self, X, w, y):
        pass


class SquaredLoss(Loss):
    def forward(self, X, w, y):
        minus=np.empty([])
        for x in X:
            minus=np.append(minus, np.squeeze(np.dot(np.transpose(w), x)))


        loss = np.mean((1/2) * (y - np.delete(minus, 0))**2, axis=0)

        if self.regularization:
            return self.regularization.forward(w)+loss
        else:
            return loss

    def backward(self, X, w, y):
        ans=np.ones(len(X), dtype=object)
        for i in range(len(X)):
            ans[i]=np.squeeze(np.dot(np.transpose(w),X[i]))

        ans=y-ans
        new_ans=np.ones(len(X), dtype=object)
        for i in range(len(X)):
            new_ans[i]=X[i] * ans[i]
        new_ans=(-1.)*np.mean(new_ans, axis=0)

        if self.regularization:
            return self.regularization.backward(w)+new_ans
        else:
            return new_ans


class HingeLoss(Loss):
    def forward(self, X, w, y):
        ans=np.empty([])
        for x in X:
            ans=np.squeeze(np.append(np.squeeze(ans), np.squeeze(np.dot(np.transpose(w), x))))
        ans=np.delete(ans, 0)
        ans = np.ones((1,len(X)))-y*ans
        ans = np.append(np.zeros((1,len(X))), ans, axis=0)
        loss = np.mean(np.amax(ans, axis=1), axis=0)
        if self.regularization:
            return self.regularization.forward(w)+loss
        else:
            return loss

    def backward(self, X, w, y):
        ans=np.zeros(np.shape(X)[0], dtype=object)
        for i in range(len(X)):
            if (1 - y[i] * np.squeeze(np.dot(np.transpose(w), X[i]))) > 0:
                ans[i]=-y[i]*X[i]
            else:
                ans[i]=np.zeros(np.shape(X)[1])

        if self.regularization:
            return self.regularization.backward(w)+np.mean(ans, axis=0)
        else:
            return np.mean(ans, axis=0)

class ZeroOneLoss(Loss):
    def forward(self, X, w, y):
        predictions = (X @ w > 0.0).astype(int) * 2 - 1
        loss = np.sum((predictions != y).astype(float)) / len(X)
        if self.regularization:
            loss += self.regularization.forward(w)
        return loss

    def backward(self, X, w, y):
        raise ValueError('No need to use this function for the homework :p')
