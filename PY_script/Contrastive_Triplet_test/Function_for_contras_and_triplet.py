from PY_script.Contrastive_Triplet_test.dependencies import patches,plt,random,torch,random_split,Image,os,glob
from configs.config import Config

random.seed(Config.seed)
torch.manual_seed(Config.seed)

def getPathsSetsByBrand(dir, val_split, test_split, total_set_size=None, min_images_per_brand=2):
    category_list = []
    brand_list = []

    for category in os.listdir(dir):
        category_path = os.path.join(dir, category)
        for brand in os.listdir(category_path):
           brand_path = os.path.join(category_path, brand)
           brand_list.append(brand_path)
        category_list.append(category_path)

    test_size = int(len(brand_list) * test_split)
    val_size = int(len(brand_list) * val_split)
    train_size = len(brand_list) - test_size - val_size

    generator = torch.Generator().manual_seed(Config.seed)
    train_subset, val_subset, test_subset = random_split(brand_list, [train_size, val_size, test_size], generator=generator)
    
    # random_split return a datase, not a list. To get the list of strings needed to pass to the Dataset we do the following:
    train_brand_list = [brand_list[i] for i in train_subset.indices]
    val_brand_list   = [brand_list[i] for i in val_subset.indices]
    test_brand_list  = [brand_list[i] for i in test_subset.indices]

    train_data_list = []
    val_data_list = []
    test_data_list = []

    if total_set_size is not None:
      images_per_brand = round(total_set_size / (len(train_brand_list) + len(val_brand_list) + len(test_brand_list)))

      print(f"Number of brands in training set: {len(train_brand_list)}")
      print(f"Number of brands in validation set: {len(val_brand_list)}")
      print(f"Number of brands in test set: {len(test_brand_list)}")
      print(f"images sampled per brand: {images_per_brand}")

      if images_per_brand < min_images_per_brand:
        # downscale the number of brands per set to guarantee min_images_per_brand

        print(f"not enough images per brand, resizing sets")

        new_total_brand_size = round(total_set_size / min_images_per_brand)
        new_val_brand_size = round(new_total_brand_size * val_split)
        new_test_brand_size = round(new_total_brand_size * test_split)
        new_train_brand_size = new_total_brand_size - new_val_brand_size - new_test_brand_size

        train_brand_list = random.sample(train_brand_list, new_train_brand_size)
        val_brand_list = random.sample(val_brand_list, new_val_brand_size)
        test_brand_list = random.sample(test_brand_list, new_test_brand_size)

        print(f"Number of brands in training set: {len(train_brand_list)}")
        print(f"Number of brands in validation set: {len(val_brand_list)}")
        print(f"Number of brands in test set: {len(test_brand_list)}")

        images_per_brand = min_images_per_brand
        print(f"new images sampled per brand: {images_per_brand}")
      
      for brand in train_brand_list:
        all_images = glob.glob(os.path.join(brand, '*.jpg'))
        if images_per_brand > len(all_images):
          print(f"images are less than {min_images_per_brand} for this brand: {brand}")
        sampled_images = random.sample(all_images, min(images_per_brand, len(all_images)))
        train_data_list.extend(sampled_images)

      for brand in val_brand_list:
        all_images = glob.glob(os.path.join(brand, '*.jpg'))
        if images_per_brand > len(all_images):
          print(f"images are less than {min_images_per_brand} for this brand: {brand}")
        sampled_images = random.sample(all_images, min(images_per_brand, len(all_images)))
        val_data_list.extend(sampled_images)

      for brand in test_brand_list:
        all_images = glob.glob(os.path.join(brand, '*.jpg'))
        if images_per_brand > len(all_images):
          print(f"images are less than {min_images_per_brand} for this brand: {brand}")
        sampled_images = random.sample(all_images, min(images_per_brand, len(all_images)))
        test_data_list.extend(sampled_images)
    else:
      for brand in train_brand_list:
        train_data_list.extend(glob.glob(os.path.join(brand, '*.jpg')))
      for brand in val_brand_list:
        val_data_list.extend(glob.glob(os.path.join(brand, '*.jpg')))
      for brand in test_brand_list:
        test_data_list.extend(glob.glob(os.path.join(brand, '*.jpg')))

    return train_data_list, val_data_list, test_data_list



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

