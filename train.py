import torch, torch.nn as nn
from model import TransformerCL
from memory import EpisodicMemory

model = TransformerCL(vocab_size=1000)
opt = torch.optim.Adam(model.parameters(), 1e-4)
memory = EpisodicMemory()

def train_task(dataset):
    for x,y in dataset:
        replay = memory.sample(32)
        batch = [(x,y)] + replay
        
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        
        loss = nn.CrossEntropyLoss()(model(xs).view(-1,1000), ys.view(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()
        memory.add(x,y)
