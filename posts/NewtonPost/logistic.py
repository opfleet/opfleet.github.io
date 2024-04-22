import torch

class LinearModel:
    def __init__(self):
        self.w = None
        self.old_w = None
    
    def score(self, X):
        '''
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

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

        if self.w is None:
            self.w = torch.rand((X.size()[1]))

        if self.old_w is None:
            self.old_w = torch.zeros(X.size()[1])

        return self.w @ X.T
    
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
    
    def loss(self, X, y):
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
        
        scores = self.score(X)

        expression = -y@torch.log(torch.sigmoid(scores)).T - (1-y)@torch.log(1-torch.sigmoid(scores)).T
        return expression / X.shape[0]
    
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
    
    def hessian(self, X, y):
        '''
        Compute the Hessian matrix. Using the formula H(w) = X.T@D@X,
        where D is the diagonal matrix with entries sig(s_k)(1-sig(s_k)).

        PARAMETERS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.

            y, torch.Tensor: the label vector. y.size() == (n,),
            where n is the number of data points. Vector of labels is
            in {0.0, 1.0}.

        RETURNS:
            hessian, torch.Tensor: the Hessian matrix, which is the matrix
            of second derivatives of loss function L. hessian.size() = (p, p).
        '''
        
        scores = self.score(X)
        D_vec = torch.sigmoid(scores)@(1-torch.sigmoid(scores))
        D = torch.diag(D_vec)
        return X.T@D@X
    
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

        w_k = self.model.w
        gradient = self.model.grad(X, y)

        self.model.w = self.model.w - alpha*gradient + beta*(self.model.w - self.model.old_w)
        self.model.old_w = w_k

class NewtonOptimizer:
    def __init__(self, model):
        self.model = model

    def step(self, X, y, alpha):
        '''
        Update the weight vector using Newton's Method, a second-order
        optimization technique.

        PARAMETERS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.

            y, torch.Tensor: the label vector. y.size() == (n,),
            where n is the number of data points. Vector of labels is
            in {0.0, 1.0}.

            alpha, torch.Tensor: first learning rate parameter.
            alpha.size() == (1,).

        RETURNS:
            N/A
        '''
        gradient = self.model.grad(X, y)
        hessian = self.model.hessian(X, y)
        self.model.w = self.model.w - (alpha * torch.linalg.inv(hessian))@gradient
