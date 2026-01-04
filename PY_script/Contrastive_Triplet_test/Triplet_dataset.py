import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import os

class DatasetTriplet(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
        # --- OTTIMIZZAZIONE ---
        # Creiamo un dizionario {label: [lista_di_percorsi]}
        # Questo serve per trovare velocemente i positivi e i negativi senza scorrere tutto ogni volta
        self.label_to_images = {}
        for img_path in image_paths:
            # Assumiamo struttura: .../BrandName/img.jpg
            # Adatta questo split se le tue cartelle sono diverse!
            label = os.path.basename(os.path.dirname(img_path))
            
            if label not in self.label_to_images:
                self.label_to_images[label] = []
            self.label_to_images[label].append(img_path)
            
        self.labels = list(self.label_to_images.keys())

    def __getitem__(self, index):
        # 1. ANCHOR (Immagine di partenza)
        anchor_path = self.image_paths[index]
        anchor_label = os.path.basename(os.path.dirname(anchor_path))
        
        # 2. POSITIVE (Stesso brand, immagine diversa)
        potential_positives = self.label_to_images[anchor_label]
        
        # Se c'è solo un'immagine per quel brand (caso limite), usiamo la stessa
        if len(potential_positives) > 1:
            while True:
                pos_path = random.choice(potential_positives)
                if pos_path != anchor_path:
                    break
        else:
            pos_path = anchor_path
        
        # 3. NEGATIVE (Brand diverso)
        while True:
            neg_label = random.choice(self.labels)
            if neg_label != anchor_label:
                break
        neg_path = random.choice(self.label_to_images[neg_label])

        # Caricamento immagini con gestione errori (se un file è corrotto non crasha tutto)
        try:
            anchor_img = Image.open(anchor_path).convert('RGB')
            pos_img = Image.open(pos_path).convert('RGB')
            neg_img = Image.open(neg_path).convert('RGB')
        except Exception as e:
            print(f"Errore caricamento: {e}. Uso immagini nere di fallback.")
            anchor_img = Image.new('RGB', (224, 224))
            pos_img = Image.new('RGB', (224, 224))
            neg_img = Image.new('RGB', (224, 224))

        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        # Ritorna le 3 immagini + la label (utile per debug)
        return anchor_img, pos_img, neg_img, anchor_label

    def __len__(self):
        return len(self.image_paths)