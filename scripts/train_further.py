import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import WeightedRandomSampler


def main():

    OLD_MODEL_PATH = "best_model_small_model.pth"
    NEW_DATASET_DIR = "karlsruhe_car_roads_7.5m_buffer"
    NUM_EPOCHS = 2
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-5  # smaller LR for fine-tuning
    VAL_SPLIT = 0.2
    MODEL_OUT = "best_small_model_finetuned_karlsruhe.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(NEW_DATASET_DIR, transform=train_transform)
    print("New dataset classes:", full_dataset.classes)

    checkpoint = torch.load(OLD_MODEL_PATH, map_location=device)
    old_classes = checkpoint["class_names"]
    print("Old model classes:", old_classes)


    num_classes = len(old_classes)

    targets = [label for _, label in full_dataset.samples]
    class_counts = Counter(targets)

    num_samples = sum(class_counts.values())

    class_weights = [
        num_samples / class_counts[i]
        for i in range(num_classes)
    ]

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("Class weights:", class_weights)

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    train_labels = [
        full_dataset.samples[idx][1]
        for idx in train_dataset.indices
    ]

    train_sample_weights = [
        class_weights[label].item()
        for label in train_labels
    ]

    sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = models.convnext_small(weights=None)

    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features,
        num_classes
    )

    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    print("Loaded pretrained model successfully")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc="Training", leave=False)

        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_bar.set_postfix(
                loss=train_loss / (total / BATCH_SIZE),
                acc=100 * correct / total
            )

        train_acc = 100 * correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_val_preds = []
        all_val_labels = []

        val_bar = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = outputs.argmax(dim=1)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                val_bar.set_postfix(
                    loss=val_loss / (total / BATCH_SIZE),
                    acc=100 * correct / total
                )

        val_acc = 100 * correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_names": old_classes
            }, MODEL_OUT)

        cm = confusion_matrix(all_val_labels, all_val_preds)

        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(
            classification_report(
                all_val_labels,
                all_val_preds,
                target_names=old_classes,
                digits=3
            )
        )

        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix – Epoch {epoch + 1}")
        plt.colorbar()

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, cm[i, j],
                    ha="center", va="center",
                    color="black" if cm[i, j] > cm.max() / 2 else "white"
                )

        ticks = range(cm.shape[0])

        labels = [i + 1 for i in ticks]

        plt.xticks(ticks, labels)
        plt.yticks(ticks, labels)

        plt.ylabel("Echter Wert")
        plt.xlabel("Klassifiziert")
        plt.tight_layout()
        plt.savefig("confusion_matrix_Karlsruhe_post_finetuning_small_model_val_acc.png")
        plt.show()

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")

    print(f" model saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
