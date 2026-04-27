from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:

    #-----------------------------------
    #Paths
    #---------------------------------
    project_root: Path= Path(__file__).resolve().parent.parent
    data_dir: Path=project_root/"data"
    train_dir: Path=project_root/"train"
    test_dir: Path=project_root/"test"

    output_dir: Path= project_root/"outputs"
    checkpoint_dir: Path = output_dir/"checkpoints"

    # ----------------------
    # Hardware
    #--------------------------

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    pin_memory: bool=True


    #------------------------------------
    #Image Settings
    #-----------------------------------
    image_size: int = 224
    num_classes: int =13


    #------------------------------------
    # Training Hyperparameters
    #-----------------------------------
    batch_size: int = 32
    epochs: int = 1
    learning_rate: float =3e-4
    momentum: float = .9
    weight_decay: float = 1e-4


    #------------------------------
    # scheduler
    #-------------------------
    scheduler_t_max: int=15
    scheduler_eta_min: float = 1e-6
    
    # ----------------------------
    # Randomness
    #----------------------------
    seed: int=42


    #----------------------------
    # Documentation/Logging
    #----------------------------
    save_best_only: bool = True
    checkpoint_name = "best_model.pth"

config = Config()