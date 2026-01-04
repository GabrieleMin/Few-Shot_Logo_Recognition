import torch

class Config:
    # 1. SETUP
    project_name = "FewShot"
    
    # Paths for saving results and checkpoints
    logs_dir = "./logs"
    checkpoints_dir = "./checkpoints"
    
    # Device configuration
    if torch.backends.mps.is_available():
     device = "mps"
    elif torch.cuda.is_available():
     device = "cuda"
    else:
     device = "cpu"
    seed = 42  # For reproducibility

    # 2. DATASET PATH
    dataset_root = "LogoDet-3K/LogoDet-3K"
    csv_index_path = "LogoDet-3K"

    # Split Ratios: 70% Train, 20% Validation 
    train_split_ratio = 0.7
    val_split_ratio = 0.2

    # 3. TRAINING HYPERPARAMETERS
    epochs = 20
    batch_size = 8
    learning_rate = 1e-5

    # 4. MODEL ARCHITECTURE
    backbone = "resnet50" 
    pretrained = True     
    embedding_dim = 128    

    # TRAINED MODEL PATH
    trained_model_path = ""

    # Prediciton threadshold used to decide if two logos are the same during inference
    prediciton_threashold = 0.5
 
    
   

    freeze_layers = 0
    # Transfer Learning Strategy
    freeze_early_layers = True
    # Unfreeze all layers after this specific epoch for fine-tuning
    unfreeze_at_epoch = 5

    # 5. LOSS FUNCTION
    margin = 0.2           # Minimal distance between different logos 
