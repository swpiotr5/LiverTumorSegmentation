import csv
import json
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

# --- Weighted Dice loss implementation ---
def weighted_dice_loss(pred, target, class_weights=None, smooth=1e-6):
    """
    Weighted Dice loss dla class imbalance
    pred: logits (B, C, H, W) - przed softmax
    target: maski (B, H, W) - wartości 0...C-1
    class_weights: tensor [C] - wagi dla każdej klasy (None = równe wagi)
    """
    # Softmax na predykcjach
    pred = F.softmax(pred, dim=1)  # [B, C, H, W]
    
    # Target do one-hot
    if target.ndim == 4:
        target = target.squeeze(1)  # [B, H, W]
    target = F.one_hot(target.long(), num_classes=pred.shape[1])  # [B, H, W, C]
    target = target.permute(0, 3, 1, 2).float()  # [B, C, H, W]
    
    # Domyślne wagi (równe)
    if class_weights is None:
        class_weights = torch.ones(pred.shape[1], device=pred.device)
    
    # Oblicz Dice per klasa
    total_loss = 0
    for c in range(pred.shape[1]):
        pred_c = pred[:, c, :, :].contiguous().view(-1)
        target_c = target[:, c, :, :].contiguous().view(-1)
        
        intersection = (pred_c * target_c).sum()
        dice_c = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        
        # Ważona strata dla klasy c
        total_loss += class_weights[c] * (1 - dice_c)
    
    # Normalizuj przez sumę wag
    return total_loss / class_weights.sum()

# --- Podział na train/test ---
train_dataset = Dataset(transformation=get_train_transform(), limit_patients=50)  # ✅ PEŁNY DATASET
dataset_size = len(train_dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
train_set, test_set = random_split(train_dataset, [train_size, test_size])

# Zapisz indeksy train/test do pliku JSON
train_indices = train_set.indices
test_indices = test_set.indices

split_info = {
    'train_indices': train_indices,
    'test_indices': test_indices,
    'train_size': len(train_indices),
    'test_size': len(test_indices),
    'total_size': dataset_size,
    'limit_patients': 50  # ✅ PEŁNY DATASET
}

with open('train_test_split.json', 'w') as f:
    json.dump(split_info, f, indent=2)
print(f"✓ Zapisano split: {len(train_indices)} train, {len(test_indices)} test indeksów do train_test_split.json")

train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

best_test_loss = float('inf')

# --- Szablon pętli treningowej ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Używane urządzenie: {device}")

# Załaduj model
model = UNet(in_channels=1, num_classes=3)
model = model.to(device)

# ⚠️ KLUCZOWE: Wagi klas dla weighted loss
# Background: 1.0 (baseline)
# Liver: 13.4x (93% / 7% ≈ 13x więcej background niż liver)
# Tumor: 930x (93% / 0.1% ≈ 930x więcej background niż tumor)
# Zmniejszam proporcje żeby nie zdominować: [1.0, 5.0, 50.0]
class_weights = torch.tensor([1.0, 5.0, 50.0], device=device)
print(f"\n⚖️ Class weights: Background={class_weights[0]:.1f}, Liver={class_weights[1]:.1f}, Tumor={class_weights[2]:.1f}")
print(f"   → Model będzie 50x bardziej penalizowany za błędy w guzach!\n")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.float().to(device)
        if images.ndim == 4 and images.shape[-1] == 1:
            images = images.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        masks = masks.long().to(device)  # ✅ ZMIANA: long() dla weighted_dice_loss

        optimizer.zero_grad()
        outputs = model(images)
        loss = weighted_dice_loss(outputs, masks, class_weights=class_weights)
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
            masks = masks.long().to(device)  # ✅ ZMIANA: long() dla weighted_dice_loss
            outputs = model(images)
            loss = weighted_dice_loss(outputs, masks, class_weights=class_weights)
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
