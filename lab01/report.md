# Lab 1 (71616)

Goal: Compare two models and measure their resource usage.

Method: Breast Cancer Wisconsin dataset, 70/30 stratified split, random_state=42. Training time is the median of 5 runs after warm-up. Inference latency is the median of 100 runs.

| Model               |   Accuracy |   Train ms |   Latency ms |   Size bytes |   Size KB |   Memory MB |
|:--------------------|-----------:|-----------:|-------------:|-------------:|----------:|------------:|
| Logistic Regression |     0.9415 |    308.496 |       0.208  |         1103 |      1.08 |      149.04 |
| Random Forest       |     0.9357 |    168.696 |       7.6478 |       290889 |    284.07 |      149.93 |

Deployment comparison:

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| Logistic Regression | Yes | Yes | Yes | No |
| Random Forest | Yes | Yes | Yes | No |

Conclusions:

1. Logistic Regression had the highest accuracy.
2. Logistic Regression had the lowest inference latency.
3. Logistic Regression used the least model storage.

Deployment results are estimates based on local measurements. Memory values include the Python process, not only the model. Cloud, Edge and Mobile compatibility also depends on the actual device.
