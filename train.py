import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path

from src.config import config
from src.model import build_model
from src.dataset import create_dataloaders
from src.engine import train_one_epoch, validate_one_epoch


def load_training_dataframe():
    """
    Expected folder structure:

    data/train/
        class_0/
        class_1/
        ...
        class_12/
    """

    records = []

    class_folders = sorted([
        folder for folder in config.train_dir.iterdir()
        if folder.is_dir()
    ])

    for label, class_folder in enumerate(class_folders):
        for image_path in class_folder.glob("*"):
            if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                records.append({
                    "image_path": image_path,
                    "label": label
                })

    return pd.DataFrame(records)


def save_checkpoint(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def main():
    print("Loading training data...")
    df = load_training_dataframe()

    train_loader, valid_loader = create_dataloaders(df)

    print("Building model...")
    model = build_model(
        num_classes=config.num_classes,
        pretrained=True
    ).to(config.device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.scheduler_t_max,
        eta_min=config.scheduler_eta_min
    )

    best_val_acc = 0.0

    print("Starting training...\n")

    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config.device
        )

        val_loss, val_acc = validate_one_epoch(
            model,
            valid_loader,
            criterion,
            config.device
        )

        scheduler.step()

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Val Acc:   {val_acc:.4f}\n"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            checkpoint_path = (
                config.checkpoint_dir /
                config.checkpoint_name
            )

            save_checkpoint(model, checkpoint_path)

            print(
                f"Saved new best model "
                f"(Val Acc: {best_val_acc:.4f})\n"
            )

    print("Training complete.")


if __name__ == "__main__":
    main()