from Contrastive_Triplet_test.dependencies import ET, Image, Dataset, random,torch,defaultdict,SEED
random.seed(SEED)
torch.manual_seed(SEED)


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

        # ERRORE DI INDENTAZIONE E IDX LABELS DA INTEGRARE
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

