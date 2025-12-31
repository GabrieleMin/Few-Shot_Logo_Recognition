import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class LogoResNet50(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True, num_of_freeze_layer=5, activation_fn=None):
        super(LogoResNet50, self).__init__()
        
        # 1. Load Pre-trained Weights
        # Initialize the model with weights pretrained on ImageNet for transfer learning
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            self.model = models.resnet50(weights=weights)
        else:
            self.model = models.resnet50(weights=None)
            
        # 2. Modify the Head (Fully Connected Layer)
        # We need to produce feature embeddings instead of class probabilities
        input_features_fc = self.model.fc.in_features # Typically 2048 for ResNet50
        
        head_layers = []
        # Project features to the desired embedding dimension (e.g., 128)
        head_layers.append(nn.Linear(input_features_fc, embedding_dim))
        
        # Add an optional activation function if provided
        if activation_fn is not None:
            head_layers.append(activation_fn)
        
        # Replace the original classifier with our custom embedding head
        self.model.fc = nn.Sequential(*head_layers)

        # 3. Freezing Management
        # Define the blocks here to access them in the freeze method.
        # This structure allows progressive freezing/unfreezing strategies
        self.blocks = [
            ['conv1', 'bn1'],   # Level 1
            ['layer1'],         # Level 2
            ['layer2'],         # Level 3
            ['layer3'],         # Level 4
            ['layer4'],         # Level 5: Entire backbone frozen
        ]

        # Apply the initial freezing configuration
        self.freeze_numer_of_layer(num_of_freeze_layer)
        
    def forward(self, x):
        return self.model(x)
    
    def freeze_numer_of_layer(self, num_of_freeze_layer):
        """
        Manages layer freezing for transfer learning strategies.
        
        Args:
            num_of_freeze_layer (int):
              0   -> All layers unlocked (Full Fine-Tuning)
              1-5 -> Progressively freezes the backbone layers from shallow to deep
        """
        
        # STEP 1: RESET. Unfreeze everything (requires_grad = True).
        # This ensures we start from a clean state before applying new constraints.
        for param in self.model.parameters():
            param.requires_grad = True

        # If num is 0, exit immediately (Full Fine-Tuning mode)
        if num_of_freeze_layer == 0:
            print("Configuration: Full Fine-Tuning (All layers are trainable)")
            return
        
        # Safety check to avoid index out of bounds
        limit = min(num_of_freeze_layer, len(self.blocks))
        
        frozen_list = []

        # STEP 2: Progressively freeze the requested blocks
        for i in range(limit):
            current_blocks = self.blocks[i]
            for block_name in current_blocks:
                # Retrieve the layer by name
                layer = getattr(self.model, block_name)
                
                # Freeze parameters for this specific block
                for param in layer.parameters():
                    param.requires_grad = False
                
                frozen_list.append(block_name)

        print(f"Freezing Level {limit}. Frozen blocks: {frozen_list}")

