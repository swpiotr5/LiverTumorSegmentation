import csv
train_losses = []
test_losses = []
import torch
import torch.nn.functional as F
from Dataset import Dataset
from img_transformations import get_train_transform
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet
import matplotlib.pyplot as plt

# --- Dice loss implementation ---
def dice_loss(pred, target, smooth=1e-6):
    """
    pred: logits (B, C, H, W) lub (B, 1, H, W) - przed softmax/sigmoid
    target: maski (B, H, W) lub (B, 1, H, W) - wartości 0...C-1 lub 0/1
    """
    # obsługa binarnej i wieloklasowej segmentacji
    if pred.shape[1] == 1:
        pred = torch.sigmoid(pred)
        if target.ndim == 3:
            target = target.unsqueeze(1)
    else:
        pred = F.softmax(pred, dim=1)
        # target: [B, H, W] lub [B, 1, H, W] -> [B, H, W]
        if target.ndim == 4:
            target = target.squeeze(1)
        # one-hot: [B, H, W, C]
        target = F.one_hot(target.long(), num_classes=pred.shape[1])
        target = target.permute(0, 3, 1, 2).float()  # [B, C, H, W]

    # flatten
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice

# --- Podział na train/test ---
train_dataset = Dataset(transformation=get_train_transform(), limit_patients=20)
dataset_size = len(train_dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
train_set, test_set = random_split(train_dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

# --- Szablon pętli treningowej ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Używane urządzenie: {device}")

# Załaduj model
model = UNet(in_channels=1, num_classes=3)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.float().to(device)
        if images.ndim == 4 and images.shape[-1] == 1:
            images = images.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        masks = masks.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_set)
    train_losses.append(epoch_loss)
    print(f"Epoka {epoch+1}/{num_epochs}, loss: {epoch_loss:.4f}")

    # --- Test loss na końcu każdej epoki ---
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.float().to(device)
            if images.ndim == 4 and images.shape[-1] == 1:
                images = images.permute(0, 3, 1, 2)
            masks = masks.float().to(device)
            outputs = model(images)
            loss = dice_loss(outputs, masks)
            test_loss += loss.item() * images.size(0)
    test_loss = test_loss / len(test_set)
    test_losses.append(test_loss)
    print(f"Test loss: {test_loss:.4f}")

    # --- Zapisz najlepszy model ---
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Zapisano nowy najlepszy model (test loss: {test_loss:.4f}) do best_model.pth")


# --- Zapisz wyniki do pliku CSV ---
with open("loss_log.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["epoch", "train_loss", "test_loss"])
    for i, (tr, te) in enumerate(zip(train_losses, test_losses)):
        writer.writerow([i+1, tr, te])

# --- Rysowanie wykresów ---
plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs+1), train_losses, label="Train loss")
plt.plot(range(1, num_epochs+1), test_losses, label="Test loss")
plt.xlabel("Epoch")
plt.ylabel("Dice loss")
plt.legend()
plt.title("Train/Test loss per epoch")
plt.savefig("loss_plot.png")
plt.show()
