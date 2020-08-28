import numpy as np
from your_code import GradientDescent


class MultiClassGradientDescent:
    def __init__(self, loss, regularization=None,
                 learning_rate=0.01, reg_param=0.05):
        self.loss = loss
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.reg_param = reg_param

        self.model = []
        self.classes = None

    def fit(self, features, targets, batch_size=None, max_iter=1000):
        learner = GradientDescent(loss=self.loss, regularization=self.regularization, learning_rate=self.learning_rate, reg_param=self.reg_param)

        self.classes = np.unique(targets)

        self.model = np.zeros((np.shape(self.classes)[0], features.shape[1] + 1))

        for index in range(np.shape(self.classes)[0]):
            learner.fit(features, np.where(targets == self.classes[index], 1, -1), batch_size=batch_size, max_iter=max_iter)
            self.model[index] = learner.model

    def predict(self, features):
        ans=np.empty([])
        for i in self.confidence(features):
            ans=np.append(ans, self.classes[np.argmax(i)])

        ans=np.delete(ans, 0)

        return ans


    def confidence(self, features):
        features = np.append(features, np.ones((np.shape(features)[0],1)), axis=1)

        ans=np.empty([])
        for x in features:
            for weight in self.model:
                ans=np.append(ans, np.dot(np.transpose(weight), x))

        ans=np.reshape((np.delete(ans, 0)), (np.shape(features)[0], np.shape(self.model)[0]))
        return ans





