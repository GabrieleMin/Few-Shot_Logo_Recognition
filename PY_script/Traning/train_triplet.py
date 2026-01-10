import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, desc=""): return iterator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from configs.config import Config
from Contrastive_Triplet_test.Function_for_contras_and_triplet import getTrainValPaths
from PY_script.Utilis_function.Implementation_ResNet50 import LogoResNet50
from PY_script.Contrastive_Triplet_test.Triplet_dataset import DatasetTriplet

def train_triplet():
 
    save_dir = os.path.join("checkpoints", "triplet_run")
    os.makedirs(save_dir, exist_ok=True)


    device = torch.device(Config.device)
    
    # 1. Dataset e Dataloader

    train_files, val_files = getTrainValPaths(
        Config.dataset_root, 
        val_split=Config.val_split_ratio,
        min_images_per_brand=2
    )

    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = DatasetTriplet(train_files, transform=train_transform)
    val_dataset = DatasetTriplet(val_files, transform=val_transform)

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=2)

    # 2. Model
    print("Model Initialization (Triplet)...")
    # Using Freeze=0 
    model = LogoResNet50(embedding_dim=Config.embedding_dim, pretrained=Config.pretrained, num_of_freeze_layer=Config.freeze_layers) 
    model = model.to(device)

    # 3. Loss e Optimizer
    # Margin 1.0
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    
    # Optimizer 
    # Using 0.00001 (1e-5) 
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
    # 4. Training Loop
    best_val_loss = float('inf')
    num_epochs = 10
    
    print(f"Starting training Triplet for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        # The dataset returns: anchor, positive, negative, label
        for anchor, positive, negative, _ in pbar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass triplo
            out_a = model(anchor)
            out_p = model(positive)
            out_n = model(negative)
            
            # Calculate Loss
            loss = criterion(out_a, out_p, out_n)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}")

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative, _ in val_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                out_a = model(anchor)
                out_p = model(positive)
                out_n = model(negative)
                loss = criterion(out_a, out_p, out_n)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"VALIDATION Epoch {epoch+1}: Loss = {avg_val_loss:.4f}")

        # Saving checkpoint
        checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        
        # Saving Best Model 
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model_triplet.pth"))
            print("New Best Triplet Model Saved!")

        print("-" * 50)

if __name__ == "__main__":
    train_triplet()