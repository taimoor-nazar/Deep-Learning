import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Map labels: malignant=1, benign=0 → {1, -1}
y = np.where(y == 0, 1, -1)

# Preprocess (scaling)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perceptron with Gradient Descent
class Perceptron:
    def __init__(self, lr=0.01, epochs=50):
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.epochs):
            losses = []
            for xi, yi in zip(X, y):
                z = np.dot(self.w, xi) + self.b
                # Logistic loss gradient
                grad_w = -yi * xi * self.sigmoid(-yi * z)
                grad_b = -yi * self.sigmoid(-yi * z)

                # Update
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

                # Logistic loss value
                loss = np.log(1 + np.exp(-yi * z))
                losses.append(loss)

            self.loss_history.append(np.mean(losses))

    def predict(self, X):
        return np.where(np.dot(X, self.w) + self.b >= 0, 1, -1)

# Train model
model = Perceptron(lr=0.01, epochs=100)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

# Plot Loss curve
plt.plot(model.loss_history, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Logistic Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()

# For decision boundary plots i am using only first two features
X2 = X[:, :2]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=42)

experiments = [
    {"lr": 0.001, "epochs": 50},
    {"lr": 0.01,  "epochs": 100},
    {"lr": 0.1,   "epochs": 200}
]

plt.figure(figsize=(12, 4))

for i, exp in enumerate(experiments, 1):
    model2 = Perceptron(lr=exp["lr"], epochs=exp["epochs"])
    model2.fit(X2_train, y2_train)

    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model2.predict(grid).reshape(xx.shape)

    plt.subplot(1, 3, i)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X2[:, 0], X2[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.title(f"lr={exp['lr']}, epochs={exp['epochs']}")

plt.tight_layout()
plt.show()

# final decision boundary
final_model2 = Perceptron(lr=0.01, epochs=100)
final_model2.fit(X2_train, y2_train)

x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = final_model2.predict(grid).reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="bwr")
plt.scatter(X2[:, 0], X2[:, 1], c=y, cmap="bwr", edgecolors="k", s=60, marker="o")
plt.title("Final Decision Boundary (lr=0.01, epochs=100)")
plt.xlabel("Mean Radius (scaled)")
plt.ylabel("Mean Texture (scaled)")
plt.show()