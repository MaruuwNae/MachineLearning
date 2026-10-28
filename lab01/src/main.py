import os
import sys
import time
import random
import platform
import joblib
import psutil
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

random.seed(42)
np.random.seed(42)

base = os.path.dirname(os.path.dirname(__file__))
folder = os.path.join(base, "results")
os.makedirs(folder, exist_ok=True)

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

process = psutil.Process()
results = []
accuracy_results = []

for name, model in models.items():
    print("\nTraining:", name)

    model.fit(X_train, y_train)

    times = []
    peak_memory = 0

    for i in range(5):
        start = time.perf_counter()
        model.fit(X_train, y_train)
        times.append((time.perf_counter() - start) * 1000)

        memory = process.memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, memory)

    train_time = np.median(times)

    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)

    sample = X_test[:1]
    model.predict(sample)

    times = []

    for i in range(100):
        start = time.perf_counter()
        model.predict(sample)
        times.append((time.perf_counter() - start) * 1000)

        memory = process.memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, memory)

    latency = np.median(times)

    filename = name.lower().replace(" ", "_") + ".joblib"
    path = os.path.join(folder, filename)
    joblib.dump(model, path)

    size_bytes = os.path.getsize(path)
    size_kb = size_bytes / 1024

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Train ms": round(train_time, 3),
        "Latency ms": round(latency, 4),
        "Size bytes": size_bytes,
        "Size KB": round(size_kb, 2),
        "Memory MB": round(peak_memory, 2)
    })

    accuracy_results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4)
    })

    print("Accuracy:", round(accuracy, 4))
    print("Training:", round(train_time, 3), "ms")
    print("Latency:", round(latency, 4), "ms")
    print("Size:", round(size_kb, 2), "KB")
    print("Memory:", round(peak_memory, 2), "MB")

df = pd.DataFrame(results)

df.to_csv(os.path.join(folder, "measurements.csv"), index=False)

pd.DataFrame(accuracy_results).to_csv(
    os.path.join(folder, "baseline_accuracy.csv"), index=False
)

budgets = {
    "Cloud": (1024, 100, 500 * 1024),
    "Edge": (1024, 50, 50 * 1024),
    "Mobile": (256, 20, 10 * 1024),
    "TinyML": (0.25, 10, 100)
}

with open(os.path.join(base, "report.md"), "w", encoding="utf-8") as f:
    f.write("# Lab 1 (71616)\n\n")

    f.write("Goal: Compare two models and measure their resource usage.\n\n")
    f.write("Method: Breast Cancer Wisconsin dataset, 70/30 stratified split, ")
    f.write("random_state=42. Training time is the median of 5 runs ")
    f.write("after warm-up. Inference latency is the median of 100 runs.\n\n")

    f.write(df.to_markdown(index=False))
    f.write("\n\nDeployment comparison:\n\n")

    f.write("| Model | Cloud | Edge | Mobile | TinyML |\n")
    f.write("|---|---|---|---|---|\n")

    for row in results:
        values = []

        for name, (memory_limit, latency_limit, size_limit) in budgets.items():
            fits = (
                row["Memory MB"] <= memory_limit
                and row["Latency ms"] <= latency_limit
                and row["Size KB"] <= size_limit
            )
            values.append("Yes" if fits else "No")

        f.write(f"| {row['Model']} | {' | '.join(values)} |\n")

    best = max(results, key=lambda r: r["Accuracy"])
    fastest = min(results, key=lambda r: r["Latency ms"])
    smallest = min(results, key=lambda r: r["Size KB"])

    f.write("\nConclusions:\n\n")
    f.write(f"1. {best['Model']} had the highest accuracy.\n")
    f.write(f"2. {fastest['Model']} had the lowest inference latency.\n")
    f.write(f"3. {smallest['Model']} used the least model storage.\n\n")

    f.write("Deployment results are estimates based on local measurements. ")
    f.write("Memory values include the Python process, not only the model. ")
    f.write("Cloud, Edge and Mobile compatibility also depends on the actual device.\n")

print("\nDone! Check lab01/results and lab01/report.md")