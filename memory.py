import random

class EpisodicMemory:
    def __init__(self, size=500):
        self.size = size
        self.buffer = []

    def add(self, x, y):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append((x, y))

    def sample(self, batch):
        return random.sample(self.buffer, min(batch, len(self.buffer)))
