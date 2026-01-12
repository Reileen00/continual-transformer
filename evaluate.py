

---

## evaluate.py
```python
import torch
from model import TransformerCL

def evaluate(model, tasks):
    accs = []
    for data in tasks:
        correct, total = 0,0
        for x,y in data:
            pred = model(x).argmax(-1)
            correct += (pred == y).sum().item()
            total += y.numel()
        accs.append(correct/total)
    return accs

print("Per-task accuracy:", evaluate(model, task_datasets))
