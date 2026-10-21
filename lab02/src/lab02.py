import time
import statistics
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

RESULTS = Path("lab02/results")
RESULTS.mkdir(parents=True, exist_ok=True)

# step 1
digits = load_digits()
X = digits.data / 16.0
Y = np.eye(10)[digits.target]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, stratify=digits.target, random_state=42)


# step 2
def init_params(W):
    rng = np.random.default_rng(42)
    return {
        "W1": rng.normal(0, 0.01, (64, W)), "b1": np.zeros(W),
        "W2": rng.normal(0, 0.01, (W, W)), "b2": np.zeros(W),
        "W3": rng.normal(0, 0.01, (W, 10)), "b3": np.zeros(10),
    }


# step 3
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(X, p):
    Z1 = X @ p["W1"] + p["b1"]
    A1 = np.maximum(Z1, 0)
    Z2 = A1 @ p["W2"] + p["b2"]
    A2 = np.maximum(Z2, 0)
    Z3 = A2 @ p["W3"] + p["b3"]
    P = softmax(Z3)
    return P, (X, Z1, A1, Z2, A2, P)


# step 4
def cross_entropy(P, Y):
    return -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))


# step 5
def backward(cache, Y, p):
    X, Z1, A1, Z2, A2, P = cache
    dZ3 = (P - Y) / X.shape[0]
    dW3 = A2.T @ dZ3
    db3 = dZ3.sum(0)
    dZ2 = (dZ3 @ p["W3"].T) * (Z2 > 0)
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(0)
    dZ1 = (dZ2 @ p["W2"].T) * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def gradient_check(p, X, Y, n=10, h=1e-5):
    _, cache = forward(X, p)
    grads = backward(cache, Y, p)
    rng = np.random.default_rng(0)
    rows = []
    checked = set()
    while len(rows) < n:
        name = str(rng.choice(["W1", "W2", "W3"]))
        idx = tuple(int(rng.integers(0, s)) for s in p[name].shape)
        if (name, idx) in checked:
            continue
        checked.add((name, idx))
        old = p[name][idx]
        p[name][idx] = old + h
        loss_plus = cross_entropy(forward(X, p)[0], Y)
        p[name][idx] = old - h
        loss_minus = cross_entropy(forward(X, p)[0], Y)
        p[name][idx] = old
        num = (loss_plus - loss_minus) / (2 * h)
        an = grads[name][idx]
        rel = abs(an - num) / max(abs(an), abs(num), 1e-12)
        rows.append((name, idx, an, num, rel))
    return rows


# step 6
def train(W, lr, epochs=50, batch=32):
    p = init_params(W)
    rng = np.random.default_rng(42)
    losses = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(X_train))
        total = 0.0
        for i in range(0, len(X_train), batch):
            idx = order[i:i + batch]
            P, cache = forward(X_train[idx], p)
            total += cross_entropy(P, Y_train[idx]) * len(idx)
            grads = backward(cache, Y_train[idx], p)
            for k in p:
                p[k] -= lr * grads[k]
        losses.append(total / len(X_train))
        print(f"W={W} lr={lr} epoch {epoch} loss {losses[-1]:.4f}")
    P_test, _ = forward(X_test, p)
    acc = np.mean(P_test.argmax(1) == Y_test.argmax(1))
    return p, losses, acc


out = []

p = init_params(16)
P, _ = forward(X_train[:32], p)
out.append(f"output shape: {P.shape}")
out.append(f"max |row sum - 1|: {np.max(np.abs(P.sum(axis=1) - 1)):.1e}")
out.append(f"untrained loss: {cross_entropy(P, Y_train[:32]):.4f}  (ln 10 = {np.log(10):.4f})")

results = {}
for W in [16, 64]:
    for lr in [0.001, 0.01, 0.1]:
        results[(W, lr)] = train(W, lr)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, W in zip(axes, [16, 64]):
    for lr in [0.001, 0.01, 0.1]:
        _, losses, acc = results[(W, lr)]
        ax.plot(range(1, 51), losses, label=f"lr={lr} (acc {acc:.1%})")
    ax.set_title(f"W = {W}")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(alpha=0.3)
axes[0].set_ylabel("train loss")
fig.tight_layout()
fig.savefig(RESULTS / "loss_curves.png", dpi=150)

out.append("\nfinal test accuracy:")
for (W, lr), (_, losses, acc) in results.items():
    out.append(f"W={W} lr={lr}: loss {losses[-1]:.4f}, accuracy {acc:.2%}")

rows = gradient_check(results[(64, 0.1)][0], X_train[:32], Y_train[:32])
passed = sum(r[4] < 1e-6 for r in rows)
out.append(f"\ngradient check (W=64, lr=0.1): {passed}/10 passed")
for name, idx, an, num, rel in rows:
    out.append(f"{name}{idx}: analytic {an:+.6e}, numeric {num:+.6e}, rel error {rel:.1e}")

out.append("\ntiming (1257 samples, median of 100 runs):")
for W in [16, 64]:
    p = init_params(W)
    _, cache = forward(X_train, p)
    tf, tb = [], []
    for _ in range(100):
        t0 = time.perf_counter()
        forward(X_train, p)
        tf.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        backward(cache, Y_train, p)
        tb.append(time.perf_counter() - t0)
    f = statistics.median(tf) * 1000
    b = statistics.median(tb) * 1000
    out.append(f"W={W}: forward {f:.4f} ms, backward {b:.4f} ms, ratio {b / f:.2f}")

text = "\n".join(out)
print(text)
(RESULTS / "results.txt").write_text(text + "\n")
