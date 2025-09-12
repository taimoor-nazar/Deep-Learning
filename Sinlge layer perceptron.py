#Implementation of single layer perceptron to learn 2-input logic gates (AND, OR, NAND, NOR)

import numpy as np
import matplotlib.pyplot as plt

#Activation function(step function)
def step(x):
    return np.where(x >= 0, 1, 0)

#Perceptron learning algorithm
def train_perceptron(X, y, lr=0.1):

    w = np.random.randn(X.shape[1]) #initialize weights randomly
    b = 1 
    converged = False

    while not converged:
        misclassifications = 0
        for xi, y_actual in zip(X, y):
            y_pred = step(np.dot(w, xi) + b)
            error = y_actual - y_pred
            if error != 0: #misclassified 
                w += lr * error * xi #update weights
                b += lr * error #update bias
                misclassifications += 1
        if misclassifications == 0:  
            converged = True

    return w, b

# Input data for 2-input logic gates
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

# Truth tables for gates with their expected outputs
truth_tables = {
    "AND":  np.array([0,0,0,1]),
    "OR":   np.array([0,1,1,1]),
    "NAND": np.array([1,1,1,0]),
    "NOR":  np.array([1,0,0,0])
}

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.ravel()

for idx, (gate, y) in enumerate(truth_tables.items()):
    
    #learn perceptron weights
    w, b = train_perceptron(X, y)
    print(f"{gate} gate learned weights: {w}, bias: {b}")
    
    #make predictions
    print("Outputs:")
    for xi in X:
        prediction = step(np.dot(w, xi) + b)
        print(prediction)
               
    # Plot decision boundary to show the output regions
    x1_vals = np.linspace(-0.5, 1.5, 200)
    x2_vals = np.linspace(-0.5, 1.5, 200)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = step(w[0]*X1 + w[1]*X2 + b)

    axs[idx].contourf(X1, X2, Z, cmap="coolwarm", alpha=0.6)
    axs[idx].scatter(X[:,0], X[:,1], c=y, edgecolors="k", s=100, cmap="coolwarm")
    axs[idx].set_title(gate)
    axs[idx].set_xlabel("x1")
    axs[idx].set_ylabel("x2")
    axs[idx].set_xticks([0,1])
    axs[idx].set_yticks([0,1])

plt.tight_layout()
plt.show()
