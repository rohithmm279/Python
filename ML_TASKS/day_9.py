import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# XOR Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

# ---------------------------------------------------
# Function to Plot Decision Boundary
# ---------------------------------------------------
def plot_decision_boundary(model, X, y, title):

    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.01),
        np.arange(y_min, y_max, 0.01)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.4)

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        s=150,
        edgecolors='black'
    )

    plt.title(title)
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.grid(True)
    plt.show()


# ===================================================
# MODEL 1 : WITHOUT HIDDEN LAYER
# ===================================================
print("=" * 50)
print("MODEL 1 : WITHOUT HIDDEN LAYER")
print("=" * 50)

model1 = MLPClassifier(
    hidden_layer_sizes=(),      # No hidden layer
    activation='logistic',      # Sigmoid
    solver='lbfgs',
    max_iter=5000,
    random_state=42
)

model1.fit(X, y)

pred1 = model1.predict(X)

print("\nPredicted Outputs:")
for inp, pred in zip(X, pred1):
    print(f"{inp} --> {pred}")

acc1 = accuracy_score(y, pred1)

print("\nActual Outputs   :", y)
print("Predicted Outputs:", pred1)
print("Accuracy         : {:.2f}%".format(acc1 * 100))

plot_decision_boundary(
    model1,
    X,
    y,
    "XOR - Without Hidden Layer"
)

# ===================================================
# MODEL 2 : WITH HIDDEN LAYER
# ===================================================
print("\n" + "=" * 50)
print("MODEL 2 : WITH HIDDEN LAYER")
print("=" * 50)

model2 = MLPClassifier(
    hidden_layer_sizes=(4,),    # 1 hidden layer, 4 neurons
    activation='logistic',      # Sigmoid
    solver='lbfgs',
    max_iter=5000,
    random_state=42
)

model2.fit(X, y)

pred2 = model2.predict(X)

print("\nPredicted Outputs:")
for inp, pred in zip(X, pred2):
    print(f"{inp} --> {pred}")

acc2 = accuracy_score(y, pred2)

print("\nActual Outputs   :", y)
print("Predicted Outputs:", pred2)
print("Accuracy         : {:.2f}%".format(acc2 * 100))

plot_decision_boundary(
    model2,
    X,
    y,
    "XOR - With Hidden Layer (4 Neurons)"
)

# ===================================================
# COMPARISON
# ===================================================
print("\n" + "=" * 50)
print("COMPARISON")
print("=" * 50)

print(f"Accuracy Without Hidden Layer : {acc1*100:.2f}%")
print(f"Accuracy With Hidden Layer    : {acc2*100:.2f}%")

if acc2 > acc1:
    print("\nObservation:")
    print("The model with a hidden layer performs better.")
    print("XOR is a non-linearly separable problem,")
    print("so a hidden layer is required to learn it correctly.")
else:
    print("\nObservation:")
    print("Both models performed similarly.")