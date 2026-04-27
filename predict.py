# predict.py

import torch
import pandas as pd
from pathlib import Path

from src.config import config
from src.model import build_model
from src.dataset import create_test_loader


def load_test_dataframe():
    """
    Expected folder structure:

    data/test/
        image_001.jpg
        image_002.jpg
        ...
    """

    records = []

    for image_path in sorted(config.test_dir.glob("*")):
        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            records.append({
                "image_path": image_path
            })

    return pd.DataFrame(records)


def load_model():
    model = build_model(
        num_classes=config.num_classes,
        pretrained=False
    )

    checkpoint_path = (
        config.checkpoint_dir /
        config.checkpoint_name
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location=config.device
    )

    model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()

    return model


def generate_predictions(model, loader):
    predictions = []

    with torch.no_grad():
        for images, filenames in loader:
            images = images.to(config.device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for filename, pred in zip(filenames, preds):
                predictions.append({
                    "filename": filename,
                    "label": int(pred.item())
                })

    return predictions


def save_submission(predictions):
    submission_df = pd.DataFrame(predictions)

    config.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        config.output_dir /
        "submission.csv"
    )

    submission_df.to_csv(
        output_path,
        index=False
    )

    print(f"Submission saved to: {output_path}")


def main():
    print("Loading test data...")
    test_df = load_test_dataframe()

    test_loader = create_test_loader(test_df)

    print("Loading trained model...")
    model = load_model()

    print("Generating predictions...")
    predictions = generate_predictions(
        model,
        test_loader
    )

    save_submission(predictions)


if __name__ == "__main__":
    main()