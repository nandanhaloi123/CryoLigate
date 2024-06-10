from dataloader import EMdata
from torch.utils.data import DataLoader
from unet import UNet3D
import time
from diffusion import *
from torch.optim import Adam
import torch.nn.functional as F
import torch


BATCH_SIZE=2
# Loading the data
dataset = EMdata()
dataset_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# data = next(iter(dataset_loader))
# print(data.size())
# print(data.size())

model = UNet3D(in_channels=1, num_classes=1)
# print("Num params: ", sum(p.numel() for p in model.parameters()))
# print(model)


def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

# Training

T = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 5 # Try more!

for epoch in range(epochs):
    data = next(iter(dataset_loader))
    optimizer.zero_grad()

    t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
    # x_noisy, noise = forward_diffusion_sample(data, t, device)
    # print(x_noisy.size())
    loss = get_loss(model, data, t)
    loss.backward()
    optimizer.step()
    print("EPOCH", epoch)


# Saving the model

import joblib
joblib.dump(model, 'model_saved.pkl')
