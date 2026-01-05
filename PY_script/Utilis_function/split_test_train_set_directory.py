import sys
import os
import shutil
import random
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from configs.config import Config

def split_dataset_by_brand(src_root, dst_root, test_split=0.10):
    """
    Splits brands into train/val and test folders without modifying the source.
    """
    random.seed(Config.seed)
    
    # Define split paths
    train_dir = os.path.join(dst_root, 'train_val')
    test_dir = os.path.join(dst_root, 'test')

    # 1. Collect all brand paths across all categories
    # Structure: src_root / category / brand / [images + xmls]
    brand_registry = []
    for category in os.listdir(src_root):
        cat_path = os.path.join(src_root, category)
        if not os.path.isdir(cat_path):
            continue
        
        for brand in os.listdir(cat_path):
            brand_path = os.path.join(cat_path, brand)
            if os.path.isdir(brand_path):
                brand_registry.append({
                    'category': category,
                    'brand_name': brand,
                    'full_path': brand_path
                })

    # 2. Shuffle and Split Brands
    random.shuffle(brand_registry)
    num_test = int(len(brand_registry) * test_split)
    
    test_brands = brand_registry[:num_test]
    train_brands = brand_registry[num_test:]

    print(f"Total Brands: {len(brand_registry)}")
    print(f"Moving {len(train_brands)} brands to {train_dir}")
    print(f"Moving {len(test_brands)} brands to {test_dir}")

    # 3. Helper function to copy folders
    def copy_brands(brand_list, destination_base):
        for item in tqdm(brand_list, desc=f"Copying to {os.path.basename(destination_base)}"):
            # Maintain category/brand structure
            dest_brand_path = os.path.join(destination_base, item['category'], item['brand_name'])
            
            # Create directories if they don't exist
            os.makedirs(dest_brand_path, exist_ok=True)
            
            # Copy all files (JPG and XML)
            for file_name in os.listdir(item['full_path']):
                src_file = os.path.join(item['full_path'], file_name)
                dst_file = os.path.join(dest_brand_path, file_name)
                
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file) # copy2 preserves metadata

    # 4. Execute Copy
    copy_brands(train_brands, train_dir)
    copy_brands(test_brands, test_dir)

    print("\nSplit Complete!")

# --- Usage ---
SOURCE_DATASET = Config.dataset_root
NEW_DESTINATION = "LogoDet-3K/LogoDet-3K-divided"

split_dataset_by_brand(SOURCE_DATASET, NEW_DESTINATION)