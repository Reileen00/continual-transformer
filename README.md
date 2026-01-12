# Continual Transformer with Episodic Memory

This project implements a Transformer-based continual learning system that learns a sequence of tasks without catastrophic forgetting using episodic memory replay.

## Motivation
Standard neural networks forget previously learned tasks when trained sequentially. This repository demonstrates how memory replay can mitigate forgetting in sequence models.

## Methods
We implement:
- Transformer encoder for sequence modeling
- Episodic memory buffer for rehearsal
- Sequential task training with replay

Tasks are trained one after another. After each task, a subset of samples is stored in memory and replayed during training on new tasks.

## Training
```bash
python train.py
