import numpy as np
import math
from your_code import HingeLoss, SquaredLoss
from your_code import L1Regularization, L2Regularization


class GradientDescent:
    def __init__(self, loss, regularization=None,
                 learning_rate=0.01, reg_param=0.05):
        self.learning_rate = learning_rate

        if regularization == 'l1':
            regularizer = L1Regularization(reg_param)
        elif regularization == 'l2':
            regularizer = L2Regularization(reg_param)
        elif regularization is None:
            regularizer = None
        else:
            raise ValueError(
                'Regularizer {} is not defined'.format(regularization))

        if loss == 'hinge':
            self.loss = HingeLoss(regularizer)
        elif loss == 'squared':
            self.loss = SquaredLoss(regularizer)
        else:
            raise ValueError('Loss function {} is not defined'.format(loss))

        self.model = None

    def fit(self, features, targets, batch_size=None, max_iter=1000):
        features = np.append(features, np.ones((np.shape(features)[0], 1)), axis=1)

        self.model = np.random.uniform(-0.1, 0.1, features.shape[1])

        loss = -9999999
        new_loss = self.loss.forward(features, self.model, targets)

        if batch_size:
            new_features = np.array_split(features[:-np.shape(features)[0] % batch_size, :], int(len(features) / batch_size))
            new_targets = np.array_split(targets[:-np.shape(features)[0] % batch_size], int(len(targets) / batch_size))

            order = np.random.shuffle(range(np.shape(new_features)[0]))

        counter = 0
        while abs(new_loss - loss) > 0 and counter < max_iter:
            new_loss = loss

            if batch_size:
                for i in order:
                    gradient = self.loss.backward(new_features[i], self.model, new_targets[i])
                    self.model = self.model-self.learning_rate * gradient

                loss = self.loss.forward(features, self.model, targets)
                np.random.shuffle(order)

            else:
                gradient = self.loss.backward(features, self.model, targets)
                self.model = self.model-self.learning_rate * gradient
                new_loss = self.loss.forward(features, self.model, targets)

            counter += 1


    def predict(self, features):
        features = np.squeeze(np.append(features, np.ones((np.shape(features)[0], 1)), axis=1))
        return np.squeeze(np.where(self.confidence(features) < 0, -1, 1))

    def confidence(self, features):
        ans=np.empty([])

        for x in features:
            ans=np.append(ans, np.squeeze(np.dot(np.transpose(self.model), x)))

        ans=np.delete(ans, 0)

        return np.squeeze(ans)
