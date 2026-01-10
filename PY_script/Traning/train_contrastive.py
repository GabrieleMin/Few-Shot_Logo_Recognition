print("Reading script train.py")
import sys
import os

# paths 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from configs.config import Config
from Contrastive_Triplet_test.Function_for_contras_and_triplet import getTrainValPaths
from PY_script.Contrastive_Triplet_test.Contrastive_dataset import DatasetContrastive
from PY_script.Utilis_function.Implementation_ResNet50 import LogoResNet50

def main():
    print(f"Starting training for: {Config.project_name}")
    
    device = torch.device(Config.device)
    print(f"Using device: {device}")

    # 1. DATA AND AUGMENTATION 
    print("Loading dataset...")
    train_files, val_files = getTrainValPaths(
        Config.dataset_root, 
        val_split=Config.val_split_ratio,
        min_images_per_brand=2
    )
    print(f"Training files: {len(train_files)}")

    # Transformations
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = DatasetContrastive(train_files, transform=transform)
    train_loader = DataLoader(train_dataset, 
                          batch_size=Config.batch_size, 
                          shuffle=True, 
                          num_workers=2, 
                          persistent_workers=True)
    
    # 2. VALIDATION DATASET 
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = DatasetContrastive(val_files, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=2)

    # 3. MODEL 
    print("Inizialitation LogoResNet50")
    freeze_layers = getattr(Config, 'freeze_layers', 0)

    model = LogoResNet50(
        embedding_dim=Config.embedding_dim,
        pretrained=Config.pretrained,
        num_of_freeze_layer=freeze_layers 
    )
    
    model = model.to(device)

    # 4. LOSS E OPTIMIZER
    criterion = nn.CosineEmbeddingLoss(margin=Config.margin)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.learning_rate)

    # 5. TRAINING LOOP 
    print("Starting training cycle")
    
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (img1, img2, label) in enumerate(train_loader):
            img1, img2, label = img1['image'].to(device), img2['image'].to(device), label.to(device)
            
            optimizer.zero_grad()
            
            out1 = model(img1)
            out2 = model(img2)
            
            target_label = label.float()
            target_label[target_label == 0] = -1
            
            loss = criterion(out1, out2, target_label)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{Config.epochs}] Batch {batch_idx} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} done! Average Loss: {avg_loss:.4f}")
        
        # STARTING VALIDATION
        model.eval() 
        val_loss = 0
        with torch.no_grad(): 
            for v_img1, v_img2, v_label in val_loader:
                v_img1, v_img2, v_label = v_img1['image'].to(device), v_img2['image'].to(device), v_label.to(device)
                
                v_out1 = model(v_img1)
                v_out2 = model(v_img2)
                
                v_target = v_label.float()
                v_target[v_target == 0] = -1
                
                loss = criterion(v_out1, v_out2, v_target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"VALIDATION Epoch {epoch+1}: Loss = {avg_val_loss:.4f}")
        print("-" * 50)
        

        # Store checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            if not os.path.exists(Config.checkpoints_dir):
                os.makedirs(Config.checkpoints_dir)
            torch.save(model.state_dict(), f"{Config.checkpoints_dir}/model_epoch_{epoch+1}.pth")
            print("Checkpoint saved!")

if __name__ == "__main__":
    main()