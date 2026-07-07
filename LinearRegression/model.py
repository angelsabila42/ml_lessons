import numpy as np

def predict (x, w, b):
    """
    Predicts the output of a linear regression model given input features, weights, and bias.

    Parameters:
    x (numpy.ndarray): Input features, shape (n_samples, n_features).
    w (numpy.ndarray): Weights of the model, shape (n_features,).
    b (float): Bias term.

    Returns:
    numpy.ndarray: Predicted output, shape (n_samples,).
    """
    return x * w + b

def loss(y_true, y_pred):
    """
    Computes the Mean Squared Error (MSE) loss between true and predicted values.

    Parameters:
    y_true (numpy.ndarray): True output values, shape (n_samples,).
    y_pred (numpy.ndarray): Predicted output values, shape (n_samples,).

    Returns:
    float: Mean Squared Error loss.
    """
    return np.mean((y_true - y_pred) ** 2)

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

w = 0
b = 0

learning_rate = 0.01
iterations = 1000

for i in range (iterations):
    #Make predictions and calculate loss
    predictions = predict(x,w,b)
    cost = loss(y, predictions)
    
    n = len(x)

    #Calculate gradients
    dw = (2/n) * np.sum((predictions - y) * x)
    db = (2/n) * np.sum(predictions - y) 

    #Update weights and bias
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    if i % 100 == 0:
        print(f"Iteration {i}: Cost {cost:.4f}, Weight {w:.4f}, Bias {b:.4f}")
   
print(f"\nFinal weight = {w:.4f}")
print(f"Final intercept = {b:.4f}")