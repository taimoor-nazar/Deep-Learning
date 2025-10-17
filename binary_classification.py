import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Load and preprocess Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Extract only setosa (0) and versicolor (1) with sepal features
setosa_versicolor_mask = (y == 0) | (y == 1)
X_filtered = X[setosa_versicolor_mask, :2]  # sepal length and width
y_binary = np.where(y[setosa_versicolor_mask] == 0, 1, -1)  # setosa=1, versicolor=-1

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

class Perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.weight_history = []
        self.bias_history = []
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        self.linear_output = np.dot(X, self.weights) + self.bias
        self.predictions = self.sigmoid(self.linear_output)
        return self.predictions
    
    def compute_loss(self, y_true):
        # Convert predictions to class labels (0 or 1 for loss calculation)
        y_pred_class = np.where(self.predictions >= 0.5, 1, -1)
        y_true_01 = np.where(y_true == -1, 0, 1)  # Convert -1,1 to 0,1 for loss
        
        epsilon = 1e-15
        predictions_clipped = np.clip(self.predictions, epsilon, 1 - epsilon)
        
        loss = -np.mean(y_true_01 * np.log(predictions_clipped) + 
                       (1 - y_true_01) * np.log(1 - predictions_clipped))
        return loss
    
    def compute_gradients(self, X, y_true):
        # Convert y_true to 0,1 format for gradient calculation
        y_true_01 = np.where(y_true == -1, 0, 1)
        
        # Gradient of loss 
        dloss_dlinear = self.predictions - y_true_01
        
        # Gradients
        dloss_dw = np.dot(X.T, dloss_dlinear) / len(X)
        dloss_db = np.mean(dloss_dlinear)
        
        return dloss_dw, dloss_db
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)  # Step function for final prediction
    
    def fit(self, X, y, plot_decision_boundary=False):
        
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
        self.weight_history.append(self.weights.copy())
        self.bias_history.append(self.bias)
        
        for epoch in range(self.n_epochs):
            # Forward pass
            self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y)
            self.loss_history.append(loss)
            
            # Compute gradients
            dw, db = self.compute_gradients(X, y)
            
            # Update weights and bias 
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Store history
            self.weight_history.append(self.weights.copy())
            self.bias_history.append(self.bias)
            
            # Plot decision boundary for selected epochs
            if plot_decision_boundary and (epoch % 20 == 0 or epoch == self.n_epochs - 1):
                self.plot_decision_boundary(X, y, epoch)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    def plot_decision_boundary(self, X, y, epoch):
        
        plt.figure(figsize=(10, 6))
        
        # Plot data points
        plt.scatter(X[y == 1, 0], X[y == 1, 1], 
                   color='red', marker='o', label='Setosa', alpha=0.7, s=50)
        plt.scatter(X[y == -1, 0], X[y == -1, 1], 
                   color='blue', marker='s', label='Versicolor', alpha=0.7, s=50)
        
        # Create decision boundary
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.5)
        
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Sepal Width (cm)')
        plt.title(f'Epoch {epoch + 1} - Loss: {self.loss_history[-1]:.4f}\n'
                 f'Weights: [{self.weights[0]:.3f}, {self.weights[1]:.3f}], Bias: {self.bias:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def evaluate(self, X, y):
       
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y, y_pred, pos_label=1, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(y, y_pred)
        }

perceptron = Perceptron(learning_rate=0.1, n_epochs=100)
perceptron.fit(X_train, y_train, plot_decision_boundary=True)

# Evaluate
metrics = perceptron.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")

# Plot loss convergence
plt.figure(figsize=(10, 6))
plt.plot(perceptron.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss (Binary Cross-Entropy)')
plt.title('Loss Convergence with Gradient Descent')
plt.grid(True, alpha=0.3)
plt.show()

# Final decision boundary
plt.figure(figsize=(12, 8))

# Plot training and test data
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
           color='darkred', marker='o', label='Setosa (Train)', s=60, alpha=0.7)
plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], 
           color='darkblue', marker='s', label='Versicolor (Train)', s=60, alpha=0.7)
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], 
           color='red', marker='^', label='Setosa (Test)', s=80, edgecolors='black')
plt.scatter(X_test[y_test == -1, 0], X_test[y_test == -1, 1], 
           color='blue', marker='v', label='Versicolor (Test)', s=80, edgecolors='black')

# Decision boundary
x_min, x_max = X_filtered[:, 0].min() - 0.5, X_filtered[:, 0].max() + 0.5
y_min, y_max = X_filtered[:, 1].min() - 0.5, X_filtered[:, 1].max() + 0.5

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                    np.linspace(y_min, y_max, 100))

Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)
plt.contour(xx, yy, Z, colors='black', linewidths=2, linestyles='--')

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title(f'Final Decision Boundary with Gradient Descent\n'
         f'Test Accuracy: {metrics["accuracy"]:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()