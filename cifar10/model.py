"""CIFAR-10 model. 자율연구에서는 주로 이 파일을 고친다."""
import torch.nn as nn


class CNN(nn.Module):
    """conv 2층 + linear 1층. (B, 3, 32, 32) 이미지 → (B, 10) logits."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 8 * 8, 10),
        )

    def forward(self, x):
        return self.net(x)
