import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def binary_cross_entropy(self, y, y_pred):
        loss = -(1 / len(y)) * np.sum(
            y*np.log(y_pred) + 
            (1-y)*np.log(1-y_pred)
        )
        return loss
    
    def predict(self, X):
        z = X @ self.w + self.b
        probabilities = self.sigmoid(z)
        predictions = np.where(
            probabilities >= 0.5,
            1,
            0
        )
        return predictions
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for i in range (self.iterations):
            z = X @ self.w + self.b
            
            #Making predictions using the sigmoid function
            y_pred = self.sigmoid(z)
            
            #Calculating the loss using binary cross-entropy
            cost = self.binary_cross_entropy(y, y_pred)
            
            #Calculating the gradients
            dw = (1/n_samples) * X.T @ (y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            #Updating weights and bias
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            if i % 100 == 0:
                print(f"Iteration {i}: Cost {cost:.4f}, Weight {[f'{w:.4f}' for w in self.w]}, Bias {self.b:.4f}")


X = np.array([
    [1, 1],
    [2, 1],
    [3, 2],
    [5, 3],
    [6, 4],
    [7, 5]
])

y = np.array([
    0,
    0,
    0,
    1,
    1,
    1
])

X_test = np.array([
    [8, 2],
    [9, 4],
    [10, 5],
    
])


model = LogisticRegression(iterations=2000)
model.fit(X, y)
model.predict(X_test)