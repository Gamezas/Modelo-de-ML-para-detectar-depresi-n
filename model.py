import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importando los datos y mezclandolos
data = pd.read_csv("./clean_data.csv")
data = data.sample(frac=1).reset_index(drop=True)

# Separando el train, test y validation
x = data.drop(columns=["Depression"]).values
y = data["Depression"].values

split = int(0.8 * len(data))

x_train = x[:split]
x_test = x[split:]

y_train = y[:split]
y_test = y[split:]

mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
std[std == 0] = 1.0

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

m, n = x_train.shape

# Funcion de la sigmoide
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# Funcion de costo
def cross_entropy(X, y, w, b):
    cost_sum = 0
    eps = 1e-15

    for i in range(m):
        z = np.dot(w, X[i]) + b
        g = sigmoid(z)
        g = np.clip(g, eps, 1 - eps)
        cost_sum += -(y[i]) * np.log(g) - (1 - y[i]) * np.log(1 - g)

    return (1 / m) * cost_sum


# Funcion de optimizacion
def grad_fun(X, y, w, b):
    grad_w = np.zeros(n)
    grad_b = 0

    for i in range (m):
        z = np.dot(w, X[i]) + b
        g = sigmoid(z)

        grad_b += (g - y[i])
        for j in range(n):
            grad_w[j] += (g - y[i]) * X[i, j]

    grad_b = (1 / m) * grad_b
    grad_w = (1 / m) * grad_w

    return grad_w, grad_b

def grad_des(X, y, alpha, iterations):
    w = np.zeros(n)
    b = 0

    for i in range(iterations):
        grad_w, grad_b = grad_fun(X, y, w, b)

        w = w - alpha * grad_w
        b = b - alpha * grad_b

        if i % 100 == 0:
            print(f"Iteracion {i}: Loss/Costo {cross_entropy(X, y, w, b)}")

    return w, b

def predict(X, w, b):
    preds = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        z = np.dot(w, X[i]) + b
        g = sigmoid(z)

        preds[i] = 1 if g >= 0.5 else 0

    return preds

lr = 0.1
iterations = 1000

w, b = grad_des(x_train, y_train, lr, iterations)

predictions = predict(x_train, w, b)
accuracy = np.mean(predictions == y_train) * 100
print(f"Accuracy del train: {accuracy}")

predictions = predict(x_test, w, b)
accuracy = np.mean(predictions == y_test) * 100
print(f"Accuracy del test: {accuracy}")