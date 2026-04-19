"""
Fine-tuning modelu UNet na pełnym datasecie LiTS.
Ładuje wagi z best_model.pth i trenuje dalej z niższym learning rate.

Użycie:
  cd src/
  python train_finetune.py
"""

import csv
import json
import random
import torch
import torch.nn.functional as F
from Dataset import Dataset
from img_transformations import get_train_transform
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet

# --- Loss functions (identyczne jak w train.py) ---

def weighted_dice_loss(pred, target, class_weights=None, smooth=1e-6):
    pred = F.softmax(pred, dim=1)
    if target.ndim == 4:
        target = target.squeeze(1)
    target = F.one_hot(target.long(), num_classes=pred.shape[1])
    target = target.permute(0, 3, 1, 2).float()

    if class_weights is None:
        class_weights = torch.ones(pred.shape[1], device=pred.device)

    total_loss = 0
    for c in range(pred.shape[1]):
        pred_c = pred[:, c, :, :].contiguous().view(-1)
        target_c = target[:, c, :, :].contiguous().view(-1)
        intersection = (pred_c * target_c).sum()
        dice_c = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        total_loss += class_weights[c] * (1 - dice_c)

    return total_loss / class_weights.sum()


def focal_loss(pred, target, alpha=None, gamma=2.0):
    pred = F.softmax(pred, dim=1)
    if target.ndim == 4:
        target = target.squeeze(1)
    target = F.one_hot(target.long(), num_classes=pred.shape[1])
    target = target.permute(0, 3, 1, 2).float()

    if alpha is None:
        alpha = torch.ones(pred.shape[1], device=pred.device)

    focal_weight = (1 - pred) ** gamma
    ce_loss = -target * torch.log(pred + 1e-8)
    focal_ce = focal_weight * ce_loss

    loss = 0
    for c in range(pred.shape[1]):
        loss += alpha[c] * focal_ce[:, c].mean()

    return loss / alpha.sum()


def combined_loss(pred, target, class_weights, alpha_dice=0.4, alpha_focal=0.6, gamma=2.5):
    dice = weighted_dice_loss(pred, target, class_weights=class_weights)
    focal = focal_loss(pred, target, alpha=class_weights, gamma=gamma)
    return alpha_dice * dice + alpha_focal * focal


# === KONFIGURACJA FINE-TUNINGU ===
PRETRAINED_PATH = None                    # None = trening od zera
LEARNING_RATE = 1e-3                      # Pełny LR dla treningu od zera
NUM_EPOCHS = 100                          # Pełny trening na pełnym datasecie
BATCH_SIZE = 8
LIMIT_PATIENTS = None                     # None = PEŁNY DATASET
TUMOR_OVERSAMPLE_WEIGHT = 5.0
NUM_WORKERS = 4

# Class weights (te same co w train.py)
CLASS_WEIGHTS_VALUES = [1.0, 10.0, 100.0]

# Hybrid loss params
ALPHA_DICE = 0.4
ALPHA_FOCAL = 0.6
GAMMA = 2.5


def main():
    train_losses = []
    test_losses = []

    # --- Dataset — split per pacjent ---
    print("Ładowanie datasetu...")
    base_dir = "../data/prepared_fine_tuning"
    all_persons = sorted([d for d in os.listdir(base_dir) if "person" in d])
    if LIMIT_PATIENTS is not None:
        all_persons = all_persons[:LIMIT_PATIENTS]

    random.seed(42)
    random.shuffle(all_persons)
    split = int(0.8 * len(all_persons))
    train_persons = all_persons[:split]
    test_persons  = all_persons[split:]

    train_set = Dataset(base_dir=base_dir, transformation=get_train_transform(),
                        person_list=train_persons)
    test_set  = Dataset(base_dir=base_dir, transformation=None,
                        person_list=test_persons)

    # Zapisz split
    split_info = {
        'train_persons': train_persons,
        'test_persons': test_persons,
        'train_size': len(train_set),
        'test_size': len(test_set),
        'total_persons': len(all_persons),
        'limit_patients': LIMIT_PATIENTS,
        'mode': 'finetune_per_patient'
    }
    with open('finetune_split.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"Pacjenci: {len(train_persons)} train, {len(test_persons)} test")
    print(f"Slice'y:  {len(train_set)} train, {len(test_set)} test")

    # --- Stratified Batch Sampling ---
    print("\nStratified Batch Sampling: skanowanie slice'ów z guzami...")
    sample_weights = []
    tumor_count = 0
    for i in range(len(train_set)):
        try:
            mask = np.load(train_set.masks[i])
            has_tumor = bool((mask == 2).any())
        except Exception:
            has_tumor = False
        weight = TUMOR_OVERSAMPLE_WEIGHT if has_tumor else 1.0
        sample_weights.append(weight)
        if has_tumor:
            tumor_count += 1

    print(f"  Slice'y z guzem: {tumor_count}/{len(train_set)} ({tumor_count/len(train_set)*100:.1f}%)")

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_set), replacement=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # --- Model + pretrained weights ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    model = UNet(in_channels=1, num_classes=3)

    if PRETRAINED_PATH and os.path.exists(PRETRAINED_PATH):
        print(f"Ładowanie pretrained wag z: {PRETRAINED_PATH}")
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location='cpu', weights_only=True))
        print("  Wagi załadowane pomyślnie!")
    else:
        print(f"Trening od zera (PRETRAINED_PATH={PRETRAINED_PATH})")

    model = model.to(device)

    class_weights = torch.tensor(CLASS_WEIGHTS_VALUES, device=device)
    print(f"\nFine-tuning config:")
    print(f"  LR: {LEARNING_RATE} (vs 1e-3 oryginalnie)")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Class weights: BG={class_weights[0]:.0f}, Liver={class_weights[1]:.0f}, Tumor={class_weights[2]:.0f}")
    print(f"  Loss: {ALPHA_DICE*100:.0f}% Dice + {ALPHA_FOCAL*100:.0f}% Focal (gamma={GAMMA})")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # LR Scheduler - zmniejsza LR gdy test loss stoi w miejscu
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_test_loss = float('inf')

    # --- Training loop ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.float().to(device)
            if images.ndim == 4 and images.shape[-1] == 1:
                images = images.permute(0, 3, 1, 2)
            masks = masks.long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks, class_weights=class_weights,
                                alpha_dice=ALPHA_DICE, alpha_focal=ALPHA_FOCAL, gamma=GAMMA)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_set)
        train_losses.append(epoch_loss)

        # --- Test loss ---
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.float().to(device)
                if images.ndim == 4 and images.shape[-1] == 1:
                    images = images.permute(0, 3, 1, 2)
                masks = masks.long().to(device)
                outputs = model(images)
                loss = combined_loss(outputs, masks, class_weights=class_weights,
                                    alpha_dice=ALPHA_DICE, alpha_focal=ALPHA_FOCAL, gamma=GAMMA)
                test_loss += loss.item() * images.size(0)
        test_loss = test_loss / len(test_set)
        test_losses.append(test_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | train: {epoch_loss:.4f} | test: {test_loss:.4f} | lr: {current_lr:.2e}")

        # LR scheduler step
        scheduler.step(test_loss)

        # Zapisz najlepszy model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "best_model_finetuned.pth")
            print(f"  -> Nowy best model! (test loss: {test_loss:.4f})")

    # --- Zapisz logi ---
    with open("finetune_loss_log.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "test_loss"])
        for i, (tr, te) in enumerate(zip(train_losses, test_losses)):
            writer.writerow([i + 1, tr, te])

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Train loss")
    plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Fine-tuning: Train/Test loss per epoch")
    plt.savefig("finetune_loss_plot.png")
    plt.show()

    print(f"\nGotowe! Best test loss: {best_test_loss:.4f}")
    print(f"Model zapisany: best_model_finetuned.pth")


if __name__ == "__main__":
    import os
    main()
