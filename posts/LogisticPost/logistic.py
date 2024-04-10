import torch

class LinearModel:
    def __init__(self):
        self.w = None
    
    def score(self, X):
        if self.w is None:
            self.w = torch.rand((X.size()[1]))

        return self.w @ X.T
    
    def predict(self, X):
        return torch.where(self.score(X) > 0, 1.0, 0.0)
    
class LogisticRegression(LinearModel):
    
    def loss(self, X, y):
        n = X.shape[0]
        sum = 0
        for i in range(n):
            expression1 = -y[i]*torch.log(torch.sigmoid(self.score(X)[i]))
            expression2 = (1 - y[i])*torch.log(1 - torch.sigmoid(self.score(X)[i]))
    
            sum += expression1-expression2
        return sum / n

    def grad(self, X, y):
        '''
        For an M, you can implement LogisticRegression.grad using a 
        for-loop. For an E, your solution should involve no explicit 
        loops. While working on a solution that avoids loops, you might 
        find it useful to at some point convert a tensor v with shape 
        (n,) into a tensor v_ with shape (n,1). The code v_ = v[:, None] 
        will perform this conversion for you.
        '''

        n = X.shape[0]
        sum = 0
        for i in range(n):
            expression = (torch.sigmoid(self.score(X)[i]) - y[i])*X[i]
            sum += expression
        return sum / n
    
class GradientDescentOptimizer:
    def __init__(self, model):
        self.model = model
    
    def step(self, X, y, alpha, beta):
        old_w = self.model.w
        loss = self.model.loss(X, y)
        self.model.w = self.model.w + alpha*loss + beta*(self.model.w - old_w)
