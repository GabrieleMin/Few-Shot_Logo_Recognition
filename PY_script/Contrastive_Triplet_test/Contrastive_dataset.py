from PY_script.Contrastive_Triplet_test.dependencies import ET, Image, Dataset, random, defaultdict,torch
from configs.config import Config

random.seed(Config.seed)
torch.manual_seed(Config.seed)

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
        
        # ERRORE DI INDENTAZIONE E IDX LABELS DA INTEGRARE
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

