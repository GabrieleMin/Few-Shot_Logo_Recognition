########################################################################################################################################
# ALCUNI BRANDS SEMBRANO ESSERE DUPLICATI EX LogoDet-3K\LogoDet-3K\Leisure\stein world-1 E LogoDet-3K\LogoDet-3K\Leisure\stein world-2 #
########################################################################################################################################

import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torchvision import transforms
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

SEED = 101



 # DATASET that takes the list of files for the dataset, 
 # builds a dictionary with label -> image paths (label is taken from the image path, not from the xml because it is faster)
 # when an item from the dataset is requested it returns a triplet (anchor, positive, negative)

 # each sample has the structure: sample = {"image": img_transformed, "labels": labels_list, "bbs": bb_list}

class DatasetTriplet(Dataset):
    def __init__(self, file_list, transform=None):
      self.file_list = file_list
      self.transform = transform

      self.label_to_indices = defaultdict(list)
      for idx, img_path in enumerate(self.file_list):
        # Extract label from path: eg. LogoDet-3K\LogoDet-3K\Clothes\panerai\21.jpg
        # Label is the second-to-last part of the path
        label = img_path.replace('\\', '/').split('/')[-2]
        self.label_to_indices[label].append(idx)

    def __len__(self):
        self.filelength =len(self.file_list)
        return self.filelength

    def load_image(self, image_path):
        xml_path = image_path.replace(".jpg", ".xml")
        img = Image.open(image_path)
        orig_w, orig_h = img.size
        img_transformed = self.transform(img)

        labels_list = []
        bb_list = []

        try:
          tree = ET.parse(xml_path)
          root = tree.getroot()
        except Exception as e:
          raise Exception(f"Failed to parse XML file: {xml_path} | Error: {e}")
        
        objects = root.findall("object")

        for obj in objects:
          label = obj.find("name").text
          bbox = obj.find("bndbox")
          xmin = int(bbox.find("xmin").text)
          ymin = int(bbox.find("ymin").text)
          xmax = int(bbox.find("xmax").text)
          ymax = int(bbox.find("ymax").text)

          # Scale bounding boxes to match the resized image
          new_w, new_h = img_transformed.shape[2], img_transformed.shape[1]
          x_scale = new_w / orig_w
          y_scale = new_h / orig_h

          bbox_scaled = {
              "xmin": int(xmin * x_scale),
              "ymin": int(ymin * y_scale),
              "xmax": int(xmax * x_scale),
              "ymax": int(ymax * y_scale)
          }

        labels_list.append(label)
        bb_list.append(bbox_scaled)

        
        return {"image": img_transformed, "labels": labels_list, "bbs": bb_list}

    def __getitem__(self,idx):
        anchor_img_path =self.file_list[idx]
        anchor_label = anchor_img_path.replace('\\', '/').split('/')[-2]

        anchor = self.load_image(anchor_img_path)

        # get Positive
        positive_indices = [i for i in self.label_to_indices[anchor_label] if i != idx]
        positive_idx = random.choice(positive_indices)
        positive_img_path = self.file_list[positive_idx]
        positive_label = positive_img_path.replace('\\', '/').split('/')[-2]

        positive = self.load_image(positive_img_path)

        # get Negative
        negative_label = random.choice([l for l in self.label_to_indices.keys() if l != anchor_label])
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative_img_path = self.file_list[negative_idx]

        negative = self.load_image(negative_img_path)

        

        return anchor, positive, negative


class DatasetContrastive(Dataset):
    def __init__(self, file_list, transform=None):
      self.file_list = file_list
      self.transform = transform

      self.label_to_indices = defaultdict(list)
      for idx, img_path in enumerate(self.file_list):
        # Extract label from path: LogoDet-3K\LogoDet-3K\Clothes\panerai\21.jpg
        # Label is the second-to-last part of the path
        label = img_path.replace('\\', '/').split('/')[-2]
        self.label_to_indices[label].append(idx)

    def __len__(self):
        self.filelength =len(self.file_list)
        return self.filelength

    def load_image(self, image_path):
        xml_path = image_path.replace(".jpg", ".xml")
        img = Image.open(image_path)
        orig_w, orig_h = img.size
        img_transformed = self.transform(img)

        labels_list = []
        bb_list = []

        try:
          tree = ET.parse(xml_path)
          root = tree.getroot()
        except Exception as e:
          raise Exception(f"Failed to parse XML file: {xml_path} | Error: {e}")
        
        objects = root.findall("object")

        for obj in objects:
          label = obj.find("name").text
          bbox = obj.find("bndbox")
          xmin = int(bbox.find("xmin").text)
          ymin = int(bbox.find("ymin").text)
          xmax = int(bbox.find("xmax").text)
          ymax = int(bbox.find("ymax").text)

          # Scale bounding boxes to match the resized image
          new_w, new_h = img_transformed.shape[2], img_transformed.shape[1]
          x_scale = new_w / orig_w
          y_scale = new_h / orig_h

          bbox_scaled = {
              "xmin": int(xmin * x_scale),
              "ymin": int(ymin * y_scale),
              "xmax": int(xmax * x_scale),
              "ymax": int(ymax * y_scale)
          }

        labels_list.append(label)
        bb_list.append(bbox_scaled)

        
        return {"image": img_transformed, "labels": labels_list, "bbs": bb_list}

    def __getitem__(self, idx):
      img_path = self.file_list[idx]
      label = img_path.replace('\\', '/').split('/')[-2]

      img1 = self.load_image(img_path)

      # decide if pair is positive or negative
      # NON SO SE è MEGLIO UN 50/50 O SEE è MEGLIO LA DISTRIBUZIONE NATURALE DEL DATASE.
      # CON LA DISTRIBUZIONE NATURALE è MOLTO PROBABILE CHE SOLO IMMAGINI NEGATIVE VENGANO ESTRATTE CHE NON CREDO SIA UN BENE PER IL TRAINING.
      is_positive_pair = random.choice([0, 1])

      if is_positive_pair:
          # sample positive
          pos_indices = [i for i in self.label_to_indices[label] if i != idx]
          if len(pos_indices) == 0:
              print("ERROR LOADING A POSITIVE MATCH FOR THE LOADED IMAGE")
              exit()
          else:
              idx2 = random.choice(pos_indices)
              img2_path = self.file_list[idx2]
              img2 = self.load_image(img2_path)
      else:
          # sample negative
          neg_label = random.choice([l for l in self.label_to_indices.keys() if l != label])
          idx2 = random.choice(self.label_to_indices[neg_label])
          img2_path = self.file_list[idx2]
          img2 = self.load_image(img2_path)

      # is_positive_pair is returned for quick access so you dont have to compare labels after loading the images
      return img1, img2, is_positive_pair


# Returns a list of paths divided into train, validation and test.
# the directory dir is scanned to find all brands which are then split into train, validation, test so that validation and test have proportion val_split, test_split
# if total_set_size is provided only that number of images are loaded so that for each brand at least min_images_per_brand images are present

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

    generator = torch.Generator().manual_seed(SEED)
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

def main():
    random.seed(SEED)
    

    dir = "LogoDet-3K/LogoDet-3K"
    test_split = 1/10
    val_split = 1/10

    train_data_list, val_data_list, test_data_list = getPathsSetsByBrand(dir, val_split, test_split, 1000, 15)

    print(f"images in the training set: {len(train_data_list)}")
    print(f"images in the validation set: {len(val_data_list)}")
    print(f"images in the test set: {len(test_data_list)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = DatasetContrastive(train_data_list, transform)
    val_dataset = DatasetContrastive(val_data_list, transform)
    test_dataset = DatasetContrastive(test_data_list, transform)

    sample_idx = random.randint(0, len(train_dataset) - 1)
    img1, img2, is_positive = train_dataset[sample_idx]

    print(f"THE MATCH IS {is_positive}")

    show_contrastive_with_bboxes(img1, img2)



    return

if __name__ == "__main__":
    main()
