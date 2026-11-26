"""Tiny Shakespeare 글자 단위 GPT: 5분 학습한 뒤 val loss(낮을수록 좋음)를 한 줄 JSON으로 출력한다.

karpathy/ng-video-lecture의 gpt.py를 따랐다. 바꾼 것: step 수 대신 5분 고정, 마지막 val loss를 val 전체로 한 번 잰다.
"""
import json
import os
import time
import urllib.request

import torch

from model import GPTLanguageModel, block_size, device

TIME_BUDGET = 300  # 학습 시간(초). 데이터 준비와 평가는 뺀다.
batch_size = 64
learning_rate = 3e-4
torch.manual_seed(1337)

# 데이터: Tiny Shakespeare 글자 단위. 앞 90% train, 뒤 10% val.
path = "data/input.txt"
if not os.path.exists(path):
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", path)
text = open(path, encoding="utf-8").read()
stoi = {ch: i for i, ch in enumerate(sorted(set(text)))}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch():
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([train_data[i : i + block_size] for i in ix])
    y = torch.stack([train_data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def val_loss():
    """val 전체를 block_size씩 겹치지 않게 잘라 평균 loss를 낸다."""
    model.eval()
    m = (len(val_data) - 1) // block_size
    x = val_data[: m * block_size].view(m, block_size)
    y = val_data[1 : m * block_size + 1].view(m, block_size)
    losses = [model(x[i : i + batch_size].to(device), y[i : i + batch_size].to(device))[1].item() * len(x[i : i + batch_size]) for i in range(0, m, batch_size)]
    model.train()
    return sum(losses) / m


model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
t0, step = time.time(), 0
while time.time() - t0 < TIME_BUDGET:
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    step += 1
    if step % 500 == 0:
        print(f"step {step} {time.time() - t0:.0f}s train_loss {loss.item():.4f}")

print(json.dumps({"val_loss": round(val_loss(), 4), "steps": step, "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)}))
