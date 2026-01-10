from PY_script.Contrastive_Triplet_test.dependencies import patches,plt,random,torch,random_split,Image,os,glob
from configs.config import Config

random.seed(Config.seed)
torch.manual_seed(Config.seed)

def getTrainValPaths(root_dir, val_split, total_set_size=None, min_images_per_brand=2):
    train_val_path = os.path.join(root_dir, 'train_val')
    train_val_brands = []

    # Collect brand folders
    if not os.path.exists(train_val_path):
        print(f"Warning: {train_val_path} not found.")
        return [], []

    for category in os.listdir(train_val_path):
        cat_path = os.path.join(train_val_path, category)
        if os.path.isdir(cat_path):
            for brand in os.listdir(cat_path):
                brand_full_path = os.path.join(cat_path, brand)
                if os.path.isdir(brand_full_path):
                    train_val_brands.append(brand_full_path)

    # Split brands into Train and Val
    val_size = int(len(train_val_brands) * val_split)
    train_size = len(train_val_brands) - val_size
    generator = torch.Generator().manual_seed(Config.seed)
    train_subset, val_subset = random_split(train_val_brands, [train_size, val_size], generator=generator)
    
    train_brand_list = [train_val_brands[i] for i in train_subset.indices]
    val_brand_list = [train_val_brands[i] for i in val_subset.indices]

    train_data_list = []
    val_data_list = []

    # Sampling Logic
    if total_set_size is not None:
        images_per_brand = round(total_set_size / len(train_val_brands))
        
        if images_per_brand < min_images_per_brand:
            print(f"Not enough images per brand ({images_per_brand}), downscaling brand sets to ensure {min_images_per_brand} images/brand.")
            
            # Calculate how many brands we can actually afford
            new_total_brand_count = round(total_set_size / min_images_per_brand)
            new_val_size = round(new_total_brand_count * val_split)
            new_train_size = new_total_brand_count - new_val_size

            train_brand_list = random.sample(train_brand_list, min(len(train_brand_list), new_train_size))
            val_brand_list = random.sample(val_brand_list, min(len(val_brand_list), new_val_size))
            images_per_brand = min_images_per_brand

        for brand in train_brand_list:
            imgs = glob.glob(os.path.join(brand, '*.jpg'))

            if len(imgs) < min_images_per_brand:
                print(f"images are less than {min_images_per_brand} for this brand: {brand} in the TRAIN set")

            train_data_list.extend(random.sample(imgs, min(images_per_brand, len(imgs))))
            
        for brand in val_brand_list:
            imgs = glob.glob(os.path.join(brand, '*.jpg'))

            if len(imgs) < min_images_per_brand:
                print(f"images are less than {min_images_per_brand} for this brand: {brand} in the VALIDATION set")
            
            val_data_list.extend(random.sample(imgs, min(images_per_brand, len(imgs))))
    else:
        for brand in train_brand_list:
            train_data_list.extend(glob.glob(os.path.join(brand, '*.jpg')))
        for brand in val_brand_list:
            val_data_list.extend(glob.glob(os.path.join(brand, '*.jpg')))

    return train_data_list, val_data_list

def getTestPaths(root_dir, total_set_size=None, min_images_per_brand=2):
    test_path = os.path.join(root_dir, 'test')
    test_brand_list = []

    # Collect brand folders
    if not os.path.exists(test_path):
        print(f"Warning: {test_path} not found.")
        return []

    for category in os.listdir(test_path):
        cat_path = os.path.join(test_path, category)
        if os.path.isdir(cat_path):
            for brand in os.listdir(cat_path):
                brand_full_path = os.path.join(cat_path, brand)
                if os.path.isdir(brand_full_path):
                    test_brand_list.append(brand_full_path)

    test_data_list = []

    # Sampling Logic
    if total_set_size is not None:
        images_per_brand = round(total_set_size / len(test_brand_list))
        
        if images_per_brand < min_images_per_brand:
            new_test_brand_count = round(total_set_size / min_images_per_brand)
            test_brand_list = random.sample(test_brand_list, min(len(test_brand_list), new_test_brand_count))
            images_per_brand = min_images_per_brand

        for brand in test_brand_list:
            imgs = glob.glob(os.path.join(brand, '*.jpg'))

            if len(imgs) < min_images_per_brand:
                print(f"images are less than {min_images_per_brand} for this brand: {brand} in the TEST set")
            
            test_data_list.extend(random.sample(imgs, min(images_per_brand, len(imgs))))
    else:
        for brand in test_brand_list:
            test_data_list.extend(glob.glob(os.path.join(brand, '*.jpg')))

    return test_data_list



def get_K_RandomImages(data_list, K=5):
  if K > len(data_list):
    raise ValueError("K cannot be larger than the size of data_list")
  return random.sample(data_list, K)

def visualizeImagesFromPathList(image_list, title="Images"):
    plt.figure(figsize=(15, 5))
    
    for i, img_path in enumerate(image_list):
        img = Image.open(img_path).convert("RGB")
        plt.subplot(1, len(image_list), i+1)
        plt.imshow(img)
        plt.title(img_path.split(os.sep)[-1])  # Show filename
        plt.axis('off')

    plt.suptitle(title)
    plt.show(block=True)

def show_triplet_with_bboxes(anchor, positive, negative):
  plt.figure(figsize=(15, 5))
  
  for i, item in enumerate([anchor, positive, negative]):
      img = item["image"]
      img = img.permute(1, 2, 0).numpy()
      img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # unnormalize
      img = img.clip(0, 1)

      ax = plt.subplot(1, 3, i+1)
      ax.imshow(img)
      plt.title(["Anchor", "Positive", "Negative"][i] + "\n" + ", ".join(item["labels"]))
      plt.axis('off')

      # Overlay bounding boxes
      for bbox in item["bbs"]:
          # Scale bounding box to resized image (224x224)
          x_scale = 224 / img.shape[1]
          y_scale = 224 / img.shape[0]
          rect = patches.Rectangle(
              (bbox["xmin"] * x_scale, bbox["ymin"] * y_scale),
              (bbox["xmax"] - bbox["xmin"]) * x_scale,
              (bbox["ymax"] - bbox["ymin"]) * y_scale,
              linewidth=2, edgecolor='r', facecolor='none'
          )
          ax.add_patch(rect)

  plt.suptitle("Sample Triplet with Bounding Boxes")
  plt.show()


def show_contrastive_with_bboxes(img1, img2):
  plt.figure(figsize=(15, 5))
  
  for i, item in enumerate([img1, img2]):
      img = item["image"]
      img = img.permute(1, 2, 0).numpy()
      img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # unnormalize
      img = img.clip(0, 1)

      ax = plt.subplot(1, 3, i+1)
      ax.imshow(img)
      plt.title(["img1", "img2"][i] + "\n" + ", ".join(item["labels"]))
      plt.axis('off')

      # Overlay bounding boxes
      for bbox in item["bbs"]:
          # Scale bounding box to resized image (224x224)
          x_scale = 224 / img.shape[1]
          y_scale = 224 / img.shape[0]
          rect = patches.Rectangle(
              (bbox["xmin"] * x_scale, bbox["ymin"] * y_scale),
              (bbox["xmax"] - bbox["xmin"]) * x_scale,
              (bbox["ymax"] - bbox["ymin"]) * y_scale,
              linewidth=2, edgecolor='r', facecolor='none'
          )
          ax.add_patch(rect)

  plt.suptitle("Sample Contrastive with Bounding Boxes")
  plt.show()

