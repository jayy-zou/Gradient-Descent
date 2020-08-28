import numpy as np

class Regularization:
    def __init__(self, reg_param=0.05):
        self.reg_param = reg_param

    def forward(self, w):
        pass

    def backward(self, w):
        pass


class L1Regularization(Regularization):
    def forward(self, w):
        return self.reg_param*np.sum([abs(ele) for ele in np.delete(w, len(w)-1)])

    def backward(self, w):
        ans=np.empty([])
        for ele in np.delete(w, np.shape(w)[0]-1):
            if ele>0:
                ans=np.append(ans, 1.)
            elif ele==0:
                ans=np.append(ans, 0.)
            else:
                ans=np.append(ans, -1.)

        return np.append(self.reg_param*np.delete(ans,0), 0)



class L2Regularization(Regularization):
    def forward(self, w):
        return 0.5*self.reg_param*np.sum(np.square(np.delete(w, np.shape(w)[0]-1)))

    def backward(self, w):
        return np.append(self.reg_param*np.delete(w, len(w)-1), 0)
