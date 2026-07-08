import numpy as np

class LinearRegression:
    def __init__(self,learning_rate = 0.01, iterations = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        
    
    def predict (self, X):
        return X @ self.w + self.b

    def loss(self, y_true, y_pred):
        """
        Computes the Mean Squared Error (MSE) loss between true and predicted values.

        Parameters:
        y_true (numpy.ndarray): True output values, shape (n_samples,).
        y_pred (numpy.ndarray): Predicted output values, shape (n_samples,).

        Returns:
        float: Mean Squared Error loss.
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for i in range (self.iterations):
            #Make predictions
            y_pred = self.predict(X)
            
            #Calculate loss
            cost = self.loss(y, y_pred)
        
            #Calculate gradients
            dw = (2/n_samples) * X.T @ (y_pred - y)
            db = (2/n_samples) * np.sum(y_pred - y) 

            #Update weights and bias
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
            
            if i % 100 == 0:
                print(f"Iteration {i}: Cost {cost:.4f}, Weight {[f'{w:.4f}' for w in self.w]}, Bias {self.b:.4f}")
        

X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

model = LinearRegression()
model.fit(X, y)