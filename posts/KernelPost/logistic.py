import torch

class LinearModel:
    def __init__(self, Xt):
        self.a = None
        self.Xt = Xt
    
    def score(self, X):
        '''
        Compute the scores for each data point in the feature matrix X. 
        The formula for s = sum(a_i * k(x, x_t,i)), for i to n, where
        x_t,i is the ith entry of the training data and k is a 
        positive-definite kernel. The variable a is the weight vector.

        If self.w is None, it is necessary to initialize a random 
        vector of weights.

        If self.old_w is None, it is necessary to initialize an
        empty vector of weights.

        PARAMETERS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.

        RETURNS:
            s, torch.Tensor: vector of scores. s.size() = (n,)
        '''

        if self.a is None:
            self.a = torch.rand((X.size()[1]))

        return self.kernel(X @ self.Xt).T@self.a
    
    def predict(self, X):
        '''
        Compute the predictions for each data point in the feature 
        matrix X. The prediction for the ith data point is either 
        0 or 1.

        PARAMETERS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.

        RETURNS:
            preds, torch.Tensor: vector of predictions in {0.0, 1.0}. 
            preds.size() = (n,)
        '''
        return torch.where(self.score(X) > 0, 1.0, 0.0)
    
class LogisticRegression(LinearModel):
    
    def loss(self, X, y, lam):
        '''
        Compute the empirical risk L(w) using the logistic loss function.

        PARAMETERS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.

            y, torch.Tensor: the label vector. y.size() == (n,),
            where n is the number of data points. Vector of labels is
            in {0.0, 1.0}.

        RETURNS:
            loss, torch.Tensor: the empirical risk of the LR model on 
            the feature matrix X, compared to the label vector y.
            loss.size() == (1,).
        '''
        s = self.score(X)
        regularizer = lam*torch.sum(torch.abs(self.a)) #FIX THIS

        y@torch.log(torch.sigmoid(s))

    
    def grad(self, X, y):
        '''
        For an M, you can implement LogisticRegression.grad using a 
        for-loop. For an E, your solution should involve no explicit 
        loops. While working on a solution that avoids loops, you might 
        find it useful to at some point convert a tensor v with shape 
        (n,) into a tensor v_ with shape (n,1). The code v_ = v[:, None] 
        will perform this conversion for you.

        Compute the gradient of the empirical risk L(w) for logistic 
        regression.

        PARAMETERS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.

            y, torch.Tensor: the label vector. y.size() == (n,),
            where n is the number of data points. Vector of labels is
            in {0.0, 1.0}.

        RETURNS:
            gradient, torch.Tensor: the gradient of the empirical risk of
            the logistic loss function on the feature matrix X and the 
            label vector y. gradient.size() == (n,).
        '''

        scores = self.score(X)
        expression = (torch.sigmoid(scores)-y)@X
        return expression / X.shape[0]
    
class GradientDescentOptimizer:
    def __init__(self, model):
        self.model = model
    
    def step(self, X, y, alpha, beta):
        '''
        Update the weight vector using gradient descent with momentum of
        one algorithmic step.

        PARAMETERS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.

            y, torch.Tensor: the label vector. y.size() == (n,),
            where n is the number of data points. Vector of labels is
            in {0.0, 1.0}.

            alpha, torch.Tensor: first learning rate parameter.
            alpha.size() == (1,).

            beta, torch.Tensor: second learning rate parameter.
            beta.size() == (1,). When beta = 0, we have a 'regular'
            gradient descent.

        RETURNS:
            N/A
        '''

        w_k = torch.clone(self.model.w)
        gradient = self.model.grad(X, y)

        self.model.w = self.model.w - alpha*gradient + beta*(self.model.w - self.model.old_w)
        self.model.old_w = torch.clone(w_k)
