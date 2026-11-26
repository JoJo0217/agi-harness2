"""CIFAR-10: 학습한 뒤 test accuracy를 한 줄 JSON으로 출력한다."""
import json

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split

from model import CNN

EPOCHS = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# 데이터: 공식 train 5만 장에서 train 5천, val 1만을 뽑고(나머지 3만 5천은 안 씀), 공식 test 1만 장을 test로 쓴다.
tf = torchvision.transforms.ToTensor()
full = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=tf)
train, val, _ = random_split(full, [5000, 10000, 35000], generator=torch.Generator().manual_seed(0))
test = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=tf)

model = CNN().to(device)


@torch.no_grad()
def accuracy(data):
    model.eval()
    correct = sum((model(x.to(device)).argmax(-1) == y.to(device)).sum().item() for x, y in DataLoader(data, batch_size=256))
    model.train()
    return correct / len(data)


opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for epoch in range(EPOCHS):
    for x, y in DataLoader(train, batch_size=64, shuffle=True):
        loss = F.cross_entropy(model(x.to(device)), y.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"epoch {epoch} train_loss {loss.item():.4f} val_acc {accuracy(val):.4f}")

print(json.dumps({"test_acc": round(accuracy(test), 4)}))
