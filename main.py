
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data / 16
y = np.eye(10)[digits.target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=digits.target, random_state=42
)

def create(w):
    return [
        np.random.normal(0, 0.01, (64, w)), np.zeros(w),
        np.random.normal(0, 0.01, (w, w)), np.zeros(w),
        np.random.normal(0, 0.01, (w, 10)), np.zeros(10)
    ]

def forward(X, p):
    W1, b1, W2, b2, W3, b3 = p
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ W2 + b2
    a2 = np.maximum(0, z2)
    z3 = a2 @ W3 + b3
    e = np.exp(z3 - z3.max(axis=1, keepdims=True))
    out = e / e.sum(axis=1, keepdims=True)
    return out, (X, z1, a1, z2, a2)

def loss(y, out):
    return -np.mean(np.sum(y * np.log(out + 1e-12), axis=1))

def backward(y, out, data, p):
    X, z1, a1, z2, a2 = data
    W1, b1, W2, b2, W3, b3 = p

    d3 = (out - y) / len(y)
    dW3 = a2.T @ d3
    db3 = d3.sum(axis=0)

    d2 = (d3 @ W3.T) * (z2 > 0)
    dW2 = a1.T @ d2
    db2 = d2.sum(axis=0)

    d1 = (d2 @ W2.T) * (z1 > 0)
    dW1 = X.T @ d1
    db1 = d1.sum(axis=0)

    return [dW1, db1, dW2, db2, dW3, db3]


os.makedirs("results", exist_ok=True)
report = open("results/report.txt", "w", encoding="utf-8")

report.write("Report\n\n")
report.write("Dataset: load_digits\n")
report.write("Train/Test split: 70/30\n")
report.write("Hidden layers: 2\n")
report.write("Epochs: 50\n")
report.write("Batch size: 32\n\n")

np.random.seed(42)
p = create(16)
out, data = forward(X_train[:32], p)

print("Shape:", out.shape)
print("Softmax:", np.allclose(out.sum(axis=1), 1, atol=1e-6))
print("Initial loss:", loss(y_train[:32], out))

report.write(f"Output shape: {out.shape}\n")
report.write(f"Initial loss: {loss(y_train[:32], out):.4f}\n\n")

grads = backward(y_train[:32], out, data, p)
rng = np.random.default_rng(42)
passed = 0

report.write("Gradient Check\n")

for i in range(10):
    layer = i % 3 * 2
    choices = np.argwhere(abs(grads[layer]) > 1e-8)
    row, col = choices[rng.integers(len(choices))]
    old = p[layer][row, col]

    p[layer][row, col] = old + 1e-5
    a = loss(y_train[:32], forward(X_train[:32], p)[0])

    p[layer][row, col] = old - 1e-5
    b = loss(y_train[:32], forward(X_train[:32], p)[0])

    p[layer][row, col] = old
    num = (a - b) / 2e-5
    grad = grads[layer][row, col]
    error = abs(num - grad) / max(1e-12, abs(num) + abs(grad))

    print("Gradient", i + 1, "error:", error)
    report.write(f"Gradient {i + 1}: {error:.8f}\n")

    if error < 1e-6:
        passed += 1

print("Gradient check:", passed, "/10")

report.write(f"Passed: {passed}/10\n")
report.write(f"Result: {'PASS' if passed >= 9 else 'FAIL'}\n\n")


report.write("Training Results\n\n")

for w in [16, 64]:
    for lr in [0.001, 0.01, 0.1]:
        np.random.seed(42)
        p = create(w)
        losses = []

        for epoch in range(50):
            order = np.random.permutation(len(X_train))

            for i in range(0, len(order), 32):
                ids = order[i:i+32]
                out, data = forward(X_train[ids], p)
                grads = backward(y_train[ids], out, data, p)

                for j in range(6):
                    p[j] -= lr * grads[j]

            out, _ = forward(X_train, p)
            value = loss(y_train, out)
            losses.append(value)

            print(w, lr, epoch + 1, round(value, 4))

        out, _ = forward(X_test, p)
        acc = np.mean(out.argmax(axis=1) == y_test.argmax(axis=1))

        print("Accuracy:", w, lr, round(acc * 100, 2), "%")

        report.write(f"Width: {w}, Learning rate: {lr}\n")
        report.write(f"Final loss: {value:.4f}\n")
        report.write(f"Test accuracy: {acc * 100:.2f}%\n\n")

        plt.plot(losses, label=f"W={w}, LR={lr}")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid()
plt.savefig("results/loss_curves.png")
plt.close()


report.write("Timing Results\n\n")

for w in [16, 64]:
    p = create(w)
    f, b = [], []

    for i in range(100):
        start = time.perf_counter()
        out, data = forward(X_test[:32], p)
        f.append(time.perf_counter() - start)

        start = time.perf_counter()
        backward(y_test[:32], out, data, p)
        b.append(time.perf_counter() - start)

    forward_time = np.median(f) * 1000
    backward_time = np.median(b) * 1000

    print("Width:", w)
    print("Forward:", forward_time, "ms")
    print("Backward:", backward_time, "ms")

    report.write(f"Width: {w}\n")
    report.write(f"Forward: {forward_time:.4f} ms\n")
    report.write(f"Backward: {backward_time:.4f} ms\n\n")


report.write("Conclusion\n")
report.write("The neural network was trained with two hidden layers.\n")
report.write("Different learning rates and hidden sizes were tested.\n")
report.write("The loss curves show the training progress.\n")
report.write("Backward propagation requires additional matrix multiplications.\n")

report.close()

print("Report saved to results/report.txt")
