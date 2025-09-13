#XOR gate impleementation using multi layer network
import numpy as np
import matplotlib.pyplot as plt

#Activation function(step function)
def step(x):
    return np.where(x >= 0, 1, 0)

def summation(x, w, b):
    return np.dot(w, x) + b

#Perceptron learning algorithm
def train_perceptron(X, y, lr=0.1):

    w = np.random.randn(X.shape[1]) #initialize weights randomly
    b = 1 
    converged = False

    while not converged:
        misclassifications = 0
        for xi, y_actual in zip(X, y):
            y_pred = step(summation(xi, w, b))
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

#training OR and NAND gates
OR_w, OR_b = train_perceptron(X, np.array([0,1,1,1]))
NAND_w, NAND_b = train_perceptron(X, np.array([1,1,1,0]))

#Generating OR and NAND outputs
OR_outputs = np.array([step(summation(xi, OR_w, OR_b)) for xi in X])
NAND_outputs = np.array([step(summation(xi, NAND_w, NAND_b)) for xi in X])

# Combining OR and NAND outputs to form XOR inputs
XOR_inputs = np.vstack((OR_outputs, NAND_outputs)).T

#Training XOR gate
XOR_w, XOR_b = train_perceptron(XOR_inputs, np.array([0,1,1,0]))

#generating XOR outputs
XOR_outputs = np.array([step(summation(xi, XOR_w, XOR_b)) for xi in XOR_inputs])

print("XOR ouputs: ", XOR_outputs)

#Visualization
x1_vals = np.linspace(-0.5, 1.5, 200)
x2_vals = np.linspace(-0.5, 1.5, 200)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# For visualization: compute XOR directly from perceptron layers
def xor_net(x1, x2):
    OR_out   = step(summation([x1, x2], OR_w, OR_b))
    NAND_out = step(summation([x1, x2], NAND_w, NAND_b))
    return step(summation([OR_out, NAND_out], XOR_w, XOR_b))

Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i,j] = xor_net(X1[i,j], X2[i,j])

plt.figure(figsize=(6,6))
plt.contourf(X1, X2, Z, cmap="coolwarm", alpha=0.6)
plt.scatter(X[:,0], X[:,1], c=[0,1,1,0], edgecolors="k", s=100, cmap="coolwarm")
plt.title("XOR Gate via Perceptron Combination")
plt.xlabel("x1")
plt.ylabel("x2")
plt.xticks([0,1])
plt.yticks([0,1])
plt.show()