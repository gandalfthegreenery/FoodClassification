# Food Image Classifier

A deep learning computer vision project that classifies food images into **13 categories** using **PyTorch** and **transfer learning with ResNet50**.

Built in a competitive Kaggle environment, this project demonstrates practical machine learning engineering skills including modular training pipelines, custom datasets, checkpointing, inference workflows, and model evaluation.

## Project Highlights

* Achieved **75% validation accuracy** on a 13-class image classification task
* Implemented **transfer learning** using pretrained ResNet50
* Built modular training and inference pipelines in Python
* Used image augmentation to improve generalization
* Added checkpoint saving for best-performing models
* Generated Kaggle-style CSV submissions for test predictions

## Tech Stack

* Python
* PyTorch
* torchvision
* pandas
* NumPy
* scikit-learn
* tqdm

## Repository Structure

```text
food-image-classifier/
├── README.md
├── requirements.txt
├── train.py
├── predict.py
├── src/
│   ├── config.py
│   ├── model.py
│   ├── dataset.py
│   └── engine.py
├── data/
├── outputs/
```

## Model Architecture

This project uses **ResNet50** pretrained on ImageNet as a feature extractor, with a custom classification head for 13 output classes.

* Backbone: ResNet50
* Transfer Learning: pretrained weights
* Final Layer: custom dense classifier
* Loss Function: CrossEntropyLoss
* Optimizer: Adam
* Scheduler: CosineAnnealingLR

## Training

To train the model:

```bash
python train.py
```

Best model checkpoints are saved to:

```text
outputs/checkpoints/
```

## Inference / Prediction

To generate predictions on test images:

```bash
python predict.py
```

Submission files are saved to:

```text
outputs/submissions/submission.csv
```

## Results

* Validation Accuracy: **75%**
* Classes: **13 food categories**
* Input Size: **224 x 224**

## What This Project Demonstrates

### Computer Vision

* Image preprocessing
* Data augmentation
* Transfer learning
* Multi-class classification

### Machine Learning Engineering

* Modular code organization
* Reproducible training setup
* Config-driven experimentation
* Checkpoint management
* Inference pipeline deployment readiness

## Future Improvements

* Hyperparameter tuning
* Confusion matrix visualization
* Grad-CAM model interpretability
* EfficientNet / ViT comparison
* Dockerized training environment

## Author

Built and refactored as part of a machine learning portfolio focused on computer vision and production-ready ML systems.
