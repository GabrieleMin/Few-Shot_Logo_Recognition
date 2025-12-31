from dependencies import SEED,ET,torch,Dataset,Image,random
random.seed(SEED)
torch.manual_seed(SEED)



class DatasetTest(Dataset):
  def __init__(self, file_list, transform=None):
      self.file_list = file_list
      self.transform = transform

  def __len__(self):
      return len(self.file_list)

  def load_image(self, image_path):
      xml_path = image_path.replace(".jpg", ".xml")
      img = Image.open(image_path)
      if self.transform:
          img = self.transform(img)

      # Parse XML
      tree = ET.parse(xml_path)
      root = tree.getroot()
      objects = root.findall("object")

      # Take first label's index only
      index_text = objects[0].find("index").text
      label_idx = int(index_text)  # Convert string to int

      return {"image": img, "label": label_idx}

  def __getitem__(self, idx):
      return self.load_image(self.file_list[idx])


