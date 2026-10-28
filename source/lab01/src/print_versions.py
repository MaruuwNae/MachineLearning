import os
import sys
import platform
import importlib.metadata

base = os.path.dirname(os.path.dirname(__file__))
folder = os.path.join(base, "results")
os.makedirs(folder, exist_ok=True)

packages = [
    "numpy", "pandas", "scikit-learn", "scipy", "matplotlib",
    "seaborn", "torch", "torchvision", "torchinfo", "thop",
    "onnx", "onnxruntime", "mlflow", "memory-profiler",
    "psutil", "codecarbon", "fastapi", "uvicorn", "pytest",
    "httpx", "locust", "requests", "pyarrow", "joblib", "tqdm"
]

lines = [
    "Python: " + sys.version,
    "OS: " + platform.platform()
]

for package in packages:
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "not installed"

    lines.append(package + ": " + version)

text = "\n".join(lines)
print(text)

with open(os.path.join(folder, "versions.txt"), "w") as f:
    f.write(text)